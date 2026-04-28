"""
Ralph 循环控制器 - 负责管理生成-评价-反馈的迭代循环
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import shutil


class LoopController:
    """迭代循环控制器"""

    def __init__(self, workspace_dir: str, max_iterations: int = 10, consecutive_passes: int = 3, patience: int = 2):
        """
        初始化循环控制器

        Args:
            workspace_dir: 工作目录
            max_iterations: 最大迭代次数
            consecutive_passes: 连续通过次数要求
            patience: 失败容忍次数（超过后发出警告）
        """
        self.workspace = Path(workspace_dir)
        self.max_iterations = max_iterations
        self.consecutive_passes = consecutive_passes
        self.patience = patience

        self.current_iteration = 0
        self.consecutive_failures = 0
        self.consecutive_passes_count = 0
        self.history: List[Dict[str, Any]] = []

        # 创建当前工作空间的 iteration 目录
        self.iteration_dir = self.workspace / f"iteration_{self.current_iteration:03d}"
        self.iteration_dir.mkdir(parents=True, exist_ok=True)

    def next_iteration(self) -> Path:
        """进入下一次迭代，返回新的迭代目录"""
        self.current_iteration += 1
        self.iteration_dir = self.workspace / f"iteration_{self.current_iteration:03d}"
        self.iteration_dir.mkdir(parents=True, exist_ok=True)
        return self.iteration_dir

    def record_iteration(self, iteration_data: Dict[str, Any]) -> None:
        """
        记录一次迭代的结果

        Args:
            iteration_data: 包含以下键的字典:
                - iteration: int
                - generator_output: str
                - metrics: dict
                - human_review: dict (可选)
                - feedback: str
                - passed: bool
                - timestamp: str
        """
        self.history.append(iteration_data)

        # 保存迭代历史到 JSON
        history_file = self.workspace / "iteration_history.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def should_stop(self) -> Dict[str, Any]:
        """
        判断是否应该停止迭代

        Returns:
            Dict with:
                - should_stop: bool
                - reason: str
                - status: "success" / "max_iterations" / "patience_exceeded" / "continue"
        """
        if self.current_iteration > self.max_iterations:
            return {
                "should_stop": True,
                "reason": f"已达到最大迭代次数 {self.max_iterations}",
                "status": "max_iterations"
            }

        if self.consecutive_failures >= self.patience:
            return {
                "should_stop": True,
                "reason": f"连续失败次数 {self.consecutive_failures} 超过容忍度 {self.patience}",
                "status": "patience_exceeded"
            }

        if self.consecutive_passes_count >= self.consecutive_passes:
            return {
                "should_stop": True,
                "reason": f"连续 {self.consecutive_passes_count} 次通过，达到收敛标准",
                "status": "success"
            }

        return {
            "should_stop": False,
            "reason": "继续迭代",
            "status": "continue",
            "current_iteration": self.current_iteration,
            "consecutive_passes": self.consecutive_passes_count,
            "consecutive_failures": self.consecutive_failures
        }

    def update_status(self, passed: bool) -> None:
        """
        更新当前迭代状态

        Args:
            passed: 此次迭代是否通过
        """
        if passed:
            self.consecutive_passes_count += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_passes_count = 0

    def get_best_iteration(self) -> Optional[Dict[str, Any]]:
        """
        获取历史中表现最好的迭代结果

        Returns:
            最佳迭代的 iteration_data 或 None
        """
        if not self.history:
            return None

        # 按 passed 优先，然后按 metrics 综合评分排序
        passed_iterations = [h for h in self.history if h.get("passed")]
        if not passed_iterations:
            return None

        # 返回最后一个通过的（通常是最优的，因为生成质量在迭代中提升）
        return passed_iterations[-1]

    def get_state(self) -> Dict[str, Any]:
        """获取当前状态快照"""
        return {
            "current_iteration": self.current_iteration,
            "consecutive_passes": self.consecutive_passes_count,
            "consecutive_failures": self.consecutive_failures,
            "max_iterations": self.max_iterations,
            "patience": self.patience,
            "history_length": len(self.history)
        }

    def save_state(self, filepath: Optional[str] = None) -> None:
        """保存状态到文件"""
        if filepath is None:
            filepath = str(self.workspace / "loop_controller_state.json")

        state = {
            **self.get_state(),
            "history": self.history,
            "saved_at": datetime.now().isoformat()
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_state(cls, filepath: str) -> "LoopController":
        """从文件加载状态"""
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)

        workspace = Path(filepath).parent
        controller = cls(
            workspace_dir=str(workspace),
            max_iterations=state.get("max_iterations", 10),
            consecutive_passes=state.get("consecutive_passes", 3),
            patience=state.get("patience", 2)
        )

        controller.current_iteration = state.get("current_iteration", 0)
        controller.consecutive_passes_count = state.get("consecutive_passes", 0)
        controller.consecutive_failures = state.get("consecutive_failures", 0)
        controller.history = state.get("history", [])

        return controller
