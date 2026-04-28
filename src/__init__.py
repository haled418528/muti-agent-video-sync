"""
Multi-Agent Video Lip-Sync
多 Agent 协作的视频口型同步训练系统
"""

__version__ = "0.1.0"
__author__ = "haled418528"

from .generator import BaseGenerator
from .evaluator import BaseEvaluator
from .ralph import LoopController

__all__ = ["BaseGenerator", "BaseEvaluator", "LoopController"]
