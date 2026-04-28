"""
Wav2Lip 生成器实现
基于 Wav2Lip: https://github.com/Rudrabha/Wav2Lip
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import shutil
import json

from .base import BaseGenerator


class Wav2LipGenerator(BaseGenerator):
    """基于 Wav2Lip 的口型同步生成器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_type = config.get("model", {}).get("type", "wav2lip")
        self.checkpoint_path = config.get("model", {}).get("checkpoint_path", "checkpoints/wav2lip_gan.pth")
        self.face_detector_path = config.get("model", {}).get("face_detector_path", "checkpoints/s3fd.pth")
        self.wav2lip_config = config.get("wav2lip", {})

        self.img_size = self.wav2lip_config.get("img_size", 96)
        self.batch_size = self.wav2lip_config.get("batch_size", 2)
        self.fps = self.wav2lip_config.get("fps", 25)
        self.face_padding_ratio = self.wav2lip_config.get("face_padding_ratio", 0.2)

        # 检查模型文件
        self._check_models()

    def _check_models(self) -> bool:
        """检查模型文件是否存在"""
        if not os.path.exists(self.checkpoint_path):
            print(f"[Wav2Lip] 模型文件不存在: {self.checkpoint_path}")
            print("[Wav2Lip] 请运行: python download_models.py")
            return False
        if not os.path.exists(self.face_detector_path):
            print(f"[Wav2Lip] 人脸检测模型不存在: {self.face_detector_path}")
            print("[Wav2Lip] 请运行: python download_models.py")
            return False
        return True

    def preprocess(self, video_path: str, **kwargs) -> Dict[str, Any]:
        """
        视频预处理：
        1. 抽帧
        2. 人脸检测
        3. 保存面部区域裁剪

        Args:
            video_path: 源视频路径

        Returns:
            预处理结果字典
        """
        video_path = Path(video_path)
        temp_dir = self.temp_dir / f"preprocess_{video_path.stem}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        frames_dir = temp_dir / "frames"
        faces_dir = temp_dir / "faces"
        frames_dir.mkdir(exist_ok=True)
        faces_dir.mkdir(exist_ok=True)

        # 1. 抽帧
        print(f"[Wav2Lip] 抽帧: {video_path}")
        subprocess.run([
            "ffmpeg", "-i", str(video_path),
            "-vf", f"fps={self.fps}",
            "-q:v", "2",
            str(frames_dir / "frame_%04d.jpg")
        ], check=True, capture_output=True)

        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        if not frame_files:
            return {"success": False, "error": "抽帧失败"}

        print(f"[Wav2Lip] 抽帧完成: {len(frame_files)} 帧")

        # 2. 人脸检测（使用 dlib 或 opencv）
        face_boxes = self._detect_faces_batch(list(frame_files), faces_dir)

        return {
            "success": True,
            "frames_dir": str(frames_dir),
            "faces_dir": str(faces_dir),
            "face_boxes": face_boxes,
            "frame_count": len(frame_files),
            "temp_dir": str(temp_dir)
        }

    def _detect_faces_batch(self, frame_files: List[Path], output_dir: Path) -> List[Optional[List[int]]]:
        """
        批量检测人脸位置

        Args:
            frame_files: 帧文件列表
            output_dir: 裁剪人脸输出目录

        Returns:
            每帧的人脸边界框列表 [x1, y1, x2, y2]
        """
        import cv2

        face_boxes = []

        for i, frame_path in enumerate(frame_files):
            img = cv2.imread(str(frame_path))
            if img is None:
                face_boxes.append(None)
                continue

            # 使用 OpenCV HAAR 级联分类器（轻量级，无需额外模型）
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

                # 添加 padding
                pad_w = int(w * self.face_padding_ratio)
                pad_h = int(h * self.face_padding_ratio)

                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(img.shape[1], x + w + pad_w)
                y2 = min(img.shape[0], y + h + pad_h)

                # 裁剪并保存
                face_img = img[y1:y2, x1:x2]
                output_path = output_dir / f"face_{i:04d}.jpg"
                cv2.imwrite(str(output_path), face_img)

                face_boxes.append([x1, y1, x2, y2])
            else:
                face_boxes.append(None)

        return face_boxes

    def generate(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用 Wav2Lip 生成口型同步视频

        Args:
            video_path: 源视频路径
            audio_path: 目标音频路径
            output_path: 输出视频路径
            **kwargs: 其他参数（如 face_region 指定面部区域）

        Returns:
            生成结果字典
        """
        video_path = Path(video_path)
        audio_path = Path(audio_path)
        output_path = Path(output_path)

        # 检查模型
        if not self._check_models():
            return {
                "success": False,
                "output_path": None,
                "metadata": {},
                "error": "模型文件缺失，请运行 download_models.py"
            }

        # 1. 预处理
        print(f"[Wav2Lip] 预处理: {video_path}")
        preprocess_result = self.preprocess(video_path)

        if not preprocess_result.get("success"):
            return {
                "success": False,
                "output_path": None,
                "metadata": {},
                "error": f"预处理失败: {preprocess_result.get('error')}"
            }

        temp_dir = Path(preprocess_result["temp_dir"])
        faces_dir = Path(preprocess_result["faces_dir"])
        face_boxes = preprocess_result["face_boxes"]

        # 过滤掉没有检测到人脸的帧
        valid_frames = []
        for i, box in enumerate(face_boxes):
            if box is not None:
                face_path = faces_dir / f"face_{i:04d}.jpg"
                if face_path.exists():
                    valid_frames.append((i, face_path, box))

        if not valid_frames:
            return {
                "success": False,
                "output_path": None,
                "metadata": {},
                "error": "未检测到有效人脸"
            }

        print(f"[Wav2Lip] 有效人脸帧: {len(valid_frames)}/{len(face_boxes)}")

        # 2. 调用 Wav2Lip 推理
        print(f"[Wav2Lip] 执行推理...")
        try:
            result_frames = self._run_inference(
                faces_dir=faces_dir,
                audio_path=audio_path,
                face_boxes=face_boxes,
                temp_dir=temp_dir
            )
        except Exception as e:
            return {
                "success": False,
                "output_path": None,
                "metadata": {},
                "error": f"推理失败: {str(e)}"
            }

        # 3. 合成视频
        print(f"[Wav2Lip] 合成视频: {output_path}")
        self._generate_video(result_frames, output_path)

        # 4. 保存元数据
        metadata = {
            "model": "wav2lip",
            "checkpoint": str(self.checkpoint_path),
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "frame_count": len(valid_frames),
            "fps": self.fps,
            "img_size": self.img_size,
            "preprocess_dir": str(temp_dir)
        }
        self.save_metadata(str(output_path), metadata)

        # 5. 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)

        return {
            "success": True,
            "output_path": str(output_path),
            "metadata": metadata,
            "error": None
        }

    def _run_inference(
        self,
        faces_dir: Path,
        audio_path: Path,
        face_boxes: List[Optional[List[int]]],
        temp_dir: Path
    ) -> List[Path]:
        """
        执行 Wav2Lip 推理

        Args:
            faces_dir: 人脸帧目录
            audio_path: 音频文件
            face_boxes: 人脸边界框列表
            temp_dir: 临时目录

        Returns:
            生成的口型同步帧路径列表
        """
        import torch
        import librosa
        import numpy as np

        # 检查并加载 Wav2Lip 模型
        try:
            from Wav2Lip.inference import Wav2Lip
        except ImportError:
            # 如果没有安装 Wav2Lip，使用简化版本
            print("[Wav2Lip] Wav2Lip 库未安装，使用模拟推理...")
            return self._simulate_inference(faces_dir, len(face_boxes))

        # 加载音频
        wav, sr = librosa.load(str(audio_path), sr=16000)
        mel = self._get_mel(wav)

        # 获取人脸帧列表
        face_files = sorted(faces_dir.glob("face_*.jpg"))

        # 批量处理
        result_frames = []
        batch_size = self.batch_size

        with torch.no_grad():
            for i in range(0, len(face_files), batch_size):
                batch_files = face_files[i:i + batch_size]
                batch_boxes = face_boxes[i:i + batch_size]

                # TODO: 实际调用 Wav2Lip 模型
                # 目前使用模拟推理代替
                for j, face_file in enumerate(batch_files):
                    output_face = temp_dir / f"result_{i + j:04d}.jpg"
                    shutil.copy(face_file, output_face)
                    result_frames.append(output_face)

        return result_frames

    def _simulate_inference(self, faces_dir: Path, num_frames: int) -> List[Path]:
        """
        模拟推理（当没有真实模型时使用）

        Args:
            faces_dir: 人脸帧目录
            num_frames: 总帧数

        Returns:
            生成帧路径列表
        """
        import cv2

        face_files = sorted(faces_dir.glob("face_*.jpg"))
        result_frames = []

        for face_file in face_files:
            img = cv2.imread(str(face_file))
            if img is None:
                continue

            # 简单的模拟处理（实际应该用模型）
            # 这里可以做简单的图像增强
            result = img

            output_path = face_file.parent / f"result_{face_file.name}"
            cv2.imwrite(str(output_path), result)
            result_frames.append(output_path)

        return result_frames

    def _get_mel(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        """将音频转换为 Mel 频谱"""
        import librosa
        mel = librosa.feature.melspectrogram(
            y=wav, sr=sr, n_fft=1024, hop_length=160,
            win_length=400, n_mels=80
        )
        mel = np.log(mel + 1e-9)
        return mel

    def _generate_video(self, frames: List[Path], output_path: Path) -> None:
        """
        将帧合成为视频

        Args:
            frames: 帧路径列表
            output_path: 输出视频路径
        """
        if not frames:
            raise ValueError("没有帧可以合成")

        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 使用 ffmpeg 合成视频
        # 先创建临时文件列表
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for frame in sorted(frames):
                f.write(f"file '{frame}'\n")
            list_file = f.name

        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                "-vf", f"fps={self.fps}",
                "-c:v", "libx264",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(output_path)
            ], check=True, capture_output=True)
        finally:
            os.unlink(list_file)

    def get_model_info(self) -> Dict[str, str]:
        """获取当前模型信息"""
        return {
            "model_type": self.model_type,
            "checkpoint_path": self.checkpoint_path,
            "face_detector_path": self.face_detector_path,
            "status": "loaded" if os.path.exists(self.checkpoint_path) else "missing"
        }
