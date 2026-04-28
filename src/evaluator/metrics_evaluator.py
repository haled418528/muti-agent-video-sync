"""
指标评价器实现
基于 SyncNet 和 LSE 指标进行口型同步质量评估
"""

import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import tempfile
import shutil

import cv2
import numpy as np

from .base import BaseEvaluator


class MetricsEvaluator(BaseEvaluator):
    """基于 SyncNet 等指标的评价器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.syncnet_model_path = config.get("model", {}).get("syncnet_path", "checkpoints/syncnet.pth")
        self.fps = config.get("wav2lip", {}).get("fps", 25)

    def evaluate(self, video_path: str, audio_path: str) -> Dict[str, Any]:
        """
        执行完整评价

        Args:
            video_path: 生成视频路径
            audio_path: 参考音频路径

        Returns:
            评价结果字典
        """
        video_path = Path(video_path)
        audio_path = Path(audio_path)

        # 计算各项指标
        metrics = {}

        # 1. SyncNet 距离
        print("[Evaluator] 计算 SyncNet 距离...")
        syncnet_dist = self.calculate_syncnet_distance(str(video_path), str(audio_path))
        metrics["syncnet_distance"] = syncnet_dist

        # 2. LSE 指标
        print("[Evaluator] 计算 LSE 指标...")
        lse = self.calculate_lse(str(video_path), str(audio_path))
        metrics["lse_d"] = lse.get("lse_d", 999.0)
        metrics["lse_c"] = lse.get("lse_c", 0.0)

        # 3. 视频质量指标
        print("[Evaluator] 计算视频质量...")
        quality = self._calculate_video_quality(str(video_path))
        metrics.update(quality)

        # 判断是否通过
        passed = self._check_pass(metrics)

        print(f"[Evaluator] 指标结果: {metrics}")
        print(f"[Evaluator] 通过: {passed}")

        return {
            "metrics": metrics,
            "passed": passed,
            "details": {
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "thresholds": {k: v.get("threshold") for k, v in self.thresholds.items()}
            }
        }

    def calculate_syncnet_distance(self, video_path: str, audio_path: str) -> float:
        """
        计算 SyncNet 距离

        SyncNet 模型用于评估口型与音频的同步程度。
        距离越小表示同步越好。

        Args:
            video_path: 视频路径
            audio_path: 音频路径

        Returns:
            SyncNet 距离值
        """
        try:
            # 提取视频中的面部区域
            face_frames = self._extract_face_frames(video_path)

            if len(face_frames) < 2:
                return 999.0

            # 加载音频特征
            import librosa
            wav, sr = librosa.load(audio_path, sr=16000)

            # 计算音频与视觉特征的时间对齐
            # 使用简单的相关性度量作为 SyncNet 的近似
            sync_score = self._compute_sync_score(face_frames, wav, sr)

            # 返回距离（1 - 相关性）
            return max(0, 1.0 - sync_score)

        except Exception as e:
            print(f"[Evaluator] SyncNet 计算失败: {e}")
            return 999.0

    def _extract_face_frames(self, video_path: str) -> List[np.ndarray]:
        """提取视频中的人脸帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 转换为灰度图进行人脸检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) > 0:
                # 取最大的人脸
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face = frame[y:y+h, x:x+w]
                # 调整大小统一
                face = cv2.resize(face, (96, 96))
                frames.append(face)

        cap.release()
        return frames

    def _compute_sync_score(
        self,
        face_frames: List[np.ndarray],
        wav: np.ndarray,
        sr: int
    ) -> float:
        """
        计算音频与视频的同步分数

        这是一个简化版本，实际的 SyncNet 使用深度学习模型。
        这里使用音频能量与口型运动的简单相关性。

        Args:
            face_frames: 人脸帧列表
            wav: 音频波形
            sr: 采样率

        Returns:
            同步分数 0-1
        """
        if len(face_frames) < 2:
            return 0.0

        # 计算口型运动（相邻帧的差异）
        prev_frame = face_frames[0]
        motions = []

        for frame in face_frames[1:]:
            # 计算灰度差异作为运动度量
            diff = np.abs(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float) -
                         cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(float))
            motions.append(np.mean(diff))
            prev_frame = frame

        motions = np.array(motions)

        # 将视频帧对应到音频时间
        frame_duration = 1.0 / self.fps
        audio_samples_per_frame = int(sr * frame_duration)

        # 计算音频能量
        audio_energies = []
        for i in range(len(motions)):
            start = i * audio_samples_per_frame
            end = min(start + audio_samples_per_frame, len(wav))
            if start < len(wav):
                audio_energies.append(np.mean(np.abs(wav[start:end])))
            else:
                audio_energies.append(0)

        audio_energies = np.array(audio_energies)

        if len(audio_energies) < 2 or len(motions) < 2:
            return 0.0

        # 计算互相关
        try:
            correlation = np.corrcoef(motions[:len(audio_energies)], audio_energies[:len(motions)])[0, 1]
            return max(0, min(1, correlation))
        except:
            return 0.5  # 默认中等同步

    def calculate_lse(self, video_path: str, audio_path: str) -> Dict[str, float]:
        """
        计算 LSE-D 和 LSE-C 指标

        LSE (Lip Sync Error) 是 Wav2Lip 官方使用的评估指标:
        - LSE-D: 距离度量，越小越好
        - LSE-C: 置信度度量，越高越好

        Args:
            video_path: 视频路径
            audio_path: 音频路径

        Returns:
            包含 lse_d 和 lse_c 的字典
        """
        try:
            # 提取特征
            face_frames = self._extract_face_frames(video_path)

            if len(face_frames) < 2:
                return {"lse_d": 999.0, "lse_c": 0.0}

            import librosa
            wav, sr = librosa.load(audio_path, sr=16000)

            # 计算 LSE-D（基于时间偏移估计）
            lse_d = self._compute_lse_d(face_frames, wav, sr)

            # 计算 LSE-C（基于同步置信度）
            lse_c = self._compute_lse_c(face_frames, wav, sr)

            return {
                "lse_d": lse_d,
                "lse_c": lse_c
            }

        except Exception as e:
            print(f"[Evaluator] LSE 计算失败: {e}")
            return {"lse_d": 999.0, "lse_c": 0.0}

    def _compute_lse_d(self, face_frames: List[np.ndarray], wav: np.ndarray, sr: int) -> float:
        """计算 LSE-D"""
        # 估计音频偏移
        offset = self._estimate_audio_offset(face_frames, wav, sr)

        # 转换为距离度量
        return abs(offset) * 100  # 缩放到合理范围

    def _compute_lse_c(self, face_frames: List[np.ndarray], wav: np.ndarray, sr: int) -> float:
        """计算 LSE-C"""
        # 计算同步置信度
        sync_score = self._compute_sync_score(face_frames, wav, sr)

        # 转换为置信度（0-10）
        return sync_score * 10

    def _estimate_audio_offset(
        self,
        face_frames: List[np.ndarray],
        wav: np.ndarray,
        sr: int
    ) -> float:
        """
        估计音频与视频之间的时间偏移（秒）

        正值表示音频超前，负值表示音频滞后
        """
        if len(face_frames) < 2:
            return 0.0

        # 计算口型运动
        motions = []
        prev = cv2.cvtColor(face_frames[0], cv2.COLOR_BGR2GRAY).astype(float)

        for frame in face_frames[1:]:
            curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
            diff = np.mean(np.abs(curr - prev))
            motions.append(diff)
            prev = curr

        motions = np.array(motions)

        # 找音频中对应的峰值位置
        frame_duration = 1.0 / self.fps
        audio_per_frame = int(sr * frame_duration)

        # 找口型运动峰值
        if len(motions) > 0:
            motion_peak_idx = np.argmax(motions)
        else:
            return 0.0

        # 计算对应的音频位置
        audio_peak_start = motion_peak_idx * audio_per_frame
        audio_peak_end = audio_peak_start + audio_per_frame

        if audio_peak_end > len(wav):
            return 0.0

        # 在音频中找对应的峰值
        audio_segment = wav[audio_peak_start:audio_peak_end]
        if len(audio_segment) < 10:
            return 0.0

        audio_peak = np.argmax(np.abs(audio_segment)) / len(audio_segment)
        motion_peak_norm = 0.5  # 假设峰值在中间

        # 计算偏移
        offset = (motion_peak_norm - audio_peak) * frame_duration
        return offset

    def _calculate_video_quality(self, video_path: str) -> Dict[str, float]:
        """
        计算视频质量指标

        Returns:
            质量指标字典
        """
        metrics = {}

        try:
            cap = cv2.VideoCapture(video_path)

            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            metrics["fps"] = fps
            metrics["frame_count"] = frame_count
            metrics["resolution"] = width * height

            # 计算帧间差异（视频稳定性）
            prev_frame = None
            frame_diffs = []
            sample_count = min(30, frame_count)

            for i in range(sample_count):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_count // sample_count)
                ret, frame = cap.read()
                if not ret:
                    break

                if prev_frame is not None:
                    diff = np.mean(np.abs(frame.astype(float) - prev_frame.astype(float)))
                    frame_diffs.append(diff)

                prev_frame = frame

            cap.release()

            if frame_diffs:
                metrics["avg_frame_diff"] = np.mean(frame_diffs)
                metrics["stability"] = 1.0 / (1.0 + np.std(frame_diffs))
            else:
                metrics["avg_frame_diff"] = 0
                metrics["stability"] = 1.0

        except Exception as e:
            print(f"[Evaluator] 视频质量计算失败: {e}")

        return metrics

    def _check_pass(self, metrics: Dict[str, float]) -> bool:
        """检查指标是否通过阈值"""
        passed_count = 0
        total_count = 0

        for metric_name, threshold_config in self.thresholds.items():
            if metric_name in metrics:
                total_count += 1
                value = metrics[metric_name]
                threshold = threshold_config["threshold"]
                direction = threshold_config.get("higher_is_better", False)

                if direction:
                    if value >= threshold:
                        passed_count += 1
                else:
                    if value <= threshold:
                        passed_count += 1

        # 如果没有 AI 指标定义，默认通过
        if total_count == 0:
            return True

        return passed_count == total_count

    def format_human_review_sample(
        self,
        video_path: str,
        audio_path: str,
        iteration: int
    ) -> Dict[str, Any]:
        """
        格式化人工评判样本

        从视频中提取关键帧和音频片段，供人工评判

        Args:
            video_path: 视频路径
            audio_path: 音频路径
            iteration: 当前迭代次数

        Returns:
            人工评判样本信息
        """
        video_path = Path(video_path)
        base_dir = video_path.parent / f"iteration_{iteration:03d}_review"
        base_dir.mkdir(parents=True, exist_ok=True)

        # 1. 提取关键帧（每秒取一帧）
        frames_dir = base_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        saved_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 每秒保存一帧
            if frame_idx % int(fps) == 0:
                output_path = frames_dir / f"frame_{frame_idx:04d}.jpg"
                cv2.imwrite(str(output_path), frame)
                saved_frames.append(str(output_path))

            frame_idx += 1

            # 最多保存 10 帧
            if len(saved_frames) >= 10:
                break

        cap.release()

        # 2. 提取音频片段（前 10 秒）
        audio_clip_path = base_dir / "audio_clip.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-t", "10",
            "-acodec", "pcm_s16le",
            str(audio_clip_path)
        ], check=True, capture_output=True)

        # 3. 生成 GIF 预览（每秒 1 帧，共 5 秒）
        gif_path = base_dir / "preview.gif"
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(video_path),
                "-vf", "fps=1,scale=320:-1",
                "-t", "5",
                "-loop", "0",
                str(gif_path)
            ], check=True, capture_output=True)
        except:
            gif_path = None

        # 4. 写入评判说明
        instructions = f"""
# 人工评判说明 - 迭代 #{iteration}

## 待评判视频
- 原始视频: {video_path}
- 目标音频: {audio_path}

## 评判维度

1. **口型同步感** (1-5分)
   - 口型动作是否与音频内容一致
   - 是否有明显的时间偏移

2. **面部自然度** (1-5分)
   - 面部是否自然，有无明显伪影
   - 口型动作是否流畅

3. **无明显伪影** (是/否)
   - 是否有闪烁、变形、模糊等问题

## 样本位置
- 关键帧: {frames_dir}
- 音频片段: {audio_clip_path}
- GIF预览: {gif_path}
- 视频: {video_path}

## 输出评判结果
请创建反馈文件: feedback.json
格式:
{{
    "iteration": {iteration},
    "lip_sync_accuracy": <1-5分数>,
    "face_naturalness": <1-5分数>,
    "no_artifact": <true/false>,
    "overall_score": <1-5分数>,
    "comments": "<其他意见>"
}}
"""

        readme_path = base_dir / "REVIEW_INSTRUCTIONS.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(instructions)

        return {
            "iteration": iteration,
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "frames_dir": str(frames_dir),
            "audio_clip_path": str(audio_clip_path),
            "gif_path": str(gif_path) if gif_path else None,
            "instructions_path": str(readme_path),
            "summary_video_path": str(gif_path) if gif_path else str(video_path)
        }

    def generate_feedback(
        self,
        metrics: Dict[str, Any],
        human_review_result: Optional[Dict] = None
    ) -> str:
        """
        生成结构化反馈

        Args:
            metrics: AI指标结果
            human_review_result: 人工评判结果

        Returns:
            格式化反馈字符串
        """
        feedback_parts = ["【AI 指标评估】"]

        # AI 指标反馈
        for metric_name, value in metrics.items():
            if metric_name in self.thresholds:
                threshold = self.thresholds[metric_name]["threshold"]
                direction = self.thresholds[metric_name].get("higher_is_better", False)

                status = "✓ 通过" if (
                    (direction and value >= threshold) or
                    (not direction and value <= threshold)
                ) else "✗ 未通过"

                feedback_parts.append(f"  - {metric_name}: {value:.3f} {status}")
            elif metric_name in ["syncnet_distance", "lse_d", "lse_c"]:
                feedback_parts.append(f"  - {metric_name}: {value:.3f}")

        # 视频质量反馈
        if "stability" in metrics:
            feedback_parts.append(f"  - 视频稳定性: {metrics['stability']:.3f}")

        # 人工评判反馈
        if human_review_result:
            feedback_parts.append("\n【人工评判】")
            for dim in self.human_review_config.get("dimensions", []):
                name = dim["name"]
                if name in human_review_result:
                    score = human_review_result[name]
                    desc = dim["description"]
                    feedback_parts.append(f"  - {desc}: {score}/5")

            if "overall_score" in human_review_result:
                feedback_parts.append(f"  - 综合评分: {human_review_result['overall_score']}/5")

        # 生成改进建议
        suggestions = self._generate_suggestions(metrics, human_review_result)
        if suggestions:
            feedback_parts.append("\n【改进建议】")
            feedback_parts.extend([f"  - {s}" for s in suggestions])

        return "\n".join(feedback_parts)

    def _generate_suggestions(
        self,
        metrics: Dict[str, Any],
        human_review_result: Optional[Dict] = None
    ) -> List[str]:
        """根据指标生成改进建议"""
        suggestions = []

        # AI 指标建议
        if metrics.get("syncnet_distance", 999) > 10:
            suggestions.append("口型同步偏差较大，建议检查音频与视频的时间对齐")

        if metrics.get("lse_c", 0) < 3:
            suggestions.append("口型置信度较低，可能需要调整模型参数或重新检测人脸")

        if metrics.get("stability", 0) < 0.5:
            suggestions.append("视频帧间变化过大，存在闪烁或不稳定问题")

        # 人工评判建议
        if human_review_result:
            if human_review_result.get("lip_sync_accuracy", 5) < 3:
                suggestions.append("口型同步感不足，建议调整音频或重新生成")

            if human_review_result.get("face_naturalness", 5) < 3:
                suggestions.append("面部自然度较差，可能存在伪影或变形问题")

            if not human_review_result.get("no_artifact", True):
                suggestions.append("存在明显伪影，建议使用后处理去伪影或调整模型")

        return suggestions
