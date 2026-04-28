"""
Generator 基类 - 定义生成器的标准接口
所有具体的生成器实现都应继承此类
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import json


class BaseGenerator(ABC):
    """生成器抽象基类"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化生成器

        Args:
            config: 配置文件字典
        """
        self.config = config
        self.temp_dir = Path(config.get("preprocessing", {}).get("temp_dir", "workspace/temp"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def generate(self, video_path: str, audio_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """
        执行口型同步生成

        Args:
            video_path: 源视频路径
            audio_path: 目标音频路径
            output_path: 输出视频路径
            **kwargs: 其他参数（如 prompt 反馈等）

        Returns:
            Dict containing:
                - success: bool
                - output_path: str
                - metadata: dict (处理信息、参数等)
                - error: str (如果失败)
        """
        pass

    @abstractmethod
    def preprocess(self, video_path: str, **kwargs) -> Dict[str, Any]:
        """
        视频预处理（抽帧、面部检测等）

        Args:
            video_path: 源视频路径
            **kwargs: 其他参数

        Returns:
            Dict containing预处理结果
        """
        pass

    def save_metadata(self, output_path: str, metadata: Dict[str, Any]) -> None:
        """保存生成元数据到同名 .json 文件"""
        meta_path = Path(output_path).with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """获取当前模型信息"""
        pass
