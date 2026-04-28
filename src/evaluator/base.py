"""
Evaluator 基类 - 定义评价器的标准接口
所有具体的评价器实现都应继承此类
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List
import json


class BaseEvaluator(ABC):
    """评价器抽象基类"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化评价器

        Args:
            config: 配置文件字典
        """
        self.config = config
        self.thresholds = config.get("ai_metrics", {})
        self.human_review_config = config.get("human_review", {})

    @abstractmethod
    def evaluate(self, video_path: str, audio_path: str) -> Dict[str, Any]:
        """
        执行评价

        Args:
            video_path: 生成视频路径
            audio_path: 参考音频路径

        Returns:
            Dict containing:
                - metrics: dict (各项AI指标)
                - passed: bool (是否通过AI评价)
                - details: dict (详细评价信息)
        """
        pass

    @abstractmethod
    def calculate_syncnet_distance(self, video_path: str, audio_path: str) -> float:
        """
        计算 SyncNet 距离

        Args:
            video_path: 视频路径
            audio_path: 音频路径

        Returns:
            SyncNet 距离值（越低越好）
        """
        pass

    @abstractmethod
    def calculate_lse(self, video_path: str, audio_path: str) -> Dict[str, float]:
        """
        计算 LSE-D 和 LSE-C 指标

        Args:
            video_path: 视频路径
            audio_path: 音频路径

        Returns:
            Dict with 'lse_d' and 'lse_c' values
        """
        pass

    def should_trigger_human_review(self, metrics: Dict[str, float]) -> bool:
        """
        判断是否需要触发人工评判

        Args:
            metrics: AI指标字典

        Returns:
            True if AI metrics pass threshold and human review should be triggered
        """
        trigger_threshold = self.human_review_config.get("trigger_threshold", 0.8)
        passed_count = 0
        total_count = len(self.thresholds)

        for metric_name, threshold_config in self.thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                threshold = threshold_config["threshold"]
                direction = "higher_is_better" if threshold_config.get("higher_is_better") else "lower_is_better"

                if direction == "higher_is_better" and value >= threshold:
                    passed_count += 1
                elif direction == "lower_is_better" and value <= threshold:
                    passed_count += 1

        pass_rate = passed_count / total_count if total_count > 0 else 0
        return pass_rate >= trigger_threshold

    @abstractmethod
    def format_human_review_sample(self, video_path: str, audio_path: str, iteration: int) -> Dict[str, Any]:
        """
        格式化人工评判样本

        Args:
            video_path: 视频路径
            audio_path: 音频路径
            iteration: 当前迭代次数

        Returns:
            Dict containing样本信息（关键帧、音频片段等）
        """
        pass

    @abstractmethod
    def generate_feedback(self, metrics: Dict[str, Any], human_review_result: Optional[Dict] = None) -> str:
        """
        生成结构化反馈

        Args:
            metrics: AI指标结果
            human_review_result: 人工评判结果（如果有）

        Returns:
            格式化的反馈字符串
        """
        pass

    def check_convergence(self, iteration_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        检查是否收敛

        Args:
            iteration_history: 每次迭代的结果历史

        Returns:
            Dict with:
                - converged: bool
                - reason: str
                - consecutive_passes: int
        """
        consecutive_passes = 0
        convergence_config = self.config.get("convergence", {})

        for i in range(len(iteration_history) - 1, -1, -1):
            if iteration_history[i].get("passed", False):
                consecutive_passes += 1
            else:
                break

        max_iterations = convergence_config.get("max_iterations", 10)
        consecutive_required = convergence_config.get("consecutive_passes", 3)

        converged = consecutive_passes >= consecutive_required

        return {
            "converged": converged,
            "reason": f"连续 {consecutive_passes}/{consecutive_required} 次通过" if converged else "未达到收敛条件",
            "consecutive_passes": consecutive_passes,
            "current_iteration": len(iteration_history),
            "max_iterations": max_iterations
        }
