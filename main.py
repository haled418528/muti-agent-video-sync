"""
Multi-Agent Video Lip-Sync - 主程序入口
通过生成器-评价器的迭代循环，不断优化口型同步效果
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# 添加 src 目录到 path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from generator.wav2lip_generator import Wav2LipGenerator
from evaluator.metrics_evaluator import MetricsEvaluator
from ralph.loop_controller import LoopController


def load_config(config_dir: Path) -> dict:
    """加载配置文件"""
    config = {}

    config_files = [
        "model_config.json",
        "eval_thresholds.json",
        "prompt_templates.json"
    ]

    for filename in config_files:
        filepath = config_dir / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                config.update(json.load(f))

    return config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Video Lip-Sync - 口型同步迭代优化系统"
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="源视频路径"
    )

    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="目标音频路径"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="输出视频路径"
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default="workspace",
        help="工作目录（存储每次迭代的结果）"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config",
        help="配置文件目录"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="最大迭代次数"
    )

    parser.add_argument(
        "--consecutive-passes",
        type=int,
        default=3,
        help="连续通过次数要求"
    )

    parser.add_argument(
        "--skip-human-review",
        action="store_true",
        help="跳过人工评判，仅使用AI指标"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 检查输入文件
    video_path = Path(args.video)
    audio_path = Path(args.audio)

    if not video_path.exists():
        print(f"错误: 视频文件不存在: {video_path}")
        sys.exit(1)

    if not audio_path.exists():
        print(f"错误: 音频文件不存在: {audio_path}")
        sys.exit(1)

    # 加载配置
    config_dir = Path(args.config)
    config = load_config(config_dir)

    # 添加命令行参数到配置
    config["skip_human_review"] = args.skip_human_review

    # 创建工作目录
    workspace_dir = Path(args.workspace)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Multi-Agent Video Lip-Sync 系统启动")
    print("=" * 60)
    print(f"源视频: {video_path}")
    print(f"目标音频: {audio_path}")
    print(f"输出路径: {args.output}")
    print(f"工作目录: {workspace_dir}")
    print(f"最大迭代: {args.max_iterations}")
    print("=" * 60)

    # 初始化组件
    print("\n[1/5] 初始化生成器...")
    generator = Wav2LipGenerator(config)

    print("[2/5] 初始化评价器...")
    evaluator = MetricsEvaluator(config)

    print("[3/5] 初始化循环控制器...")
    loop_controller = LoopController(
        workspace_dir=str(workspace_dir),
        max_iterations=args.max_iterations,
        consecutive_passes=args.consecutive_passes
    )

    # TODO: 后续集成 Claude Code Team
    # print("[4/5] 连接 Agent A (Generator) 和 Agent B (Evaluator)...")
    # team_connection = setup_team_connection()

    print("[4/5] 预留 Agent 扩展槽位...")

    print("[5/5] 开始迭代循环...")

    # 检查点路径
    checkpoint_path = workspace_dir / "last_output.mp4"

    # 迭代循环
    while True:
        iteration_dir = loop_controller.next_iteration()
        print(f"\n{'='*60}")
        print(f"迭代 #{loop_controller.current_iteration}")
        print(f"{'='*60}")

        # 准备输出路径
        output_path = iteration_dir / f"generated_{loop_controller.current_iteration:03d}.mp4"

        # 获取之前的反馈（如果有）
        feedback = None
        if len(loop_controller.history) > 0:
            last_iteration = loop_controller.history[-1]
            if "feedback" in last_iteration:
                feedback = last_iteration["feedback"]
                print(f"\n[Generator] 接收反馈:\n{feedback}")

        # ========== Agent A: 生成 ==========
        print("\n[Agent A] 执行口型同步生成...")

        if feedback:
            # TODO: 根据反馈修改提示词重新生成
            print("[Agent A] 根据反馈调整生成参数...")
        else:
            print("[Agent A] 首次生成...")

        generator_result = generator.generate(
            video_path=str(video_path),
            audio_path=str(audio_path),
            output_path=str(output_path),
            feedback=feedback
        )

        if not generator_result.get("success", False):
            print(f"[Agent A] 生成失败: {generator_result.get('error', '未知错误')}")
            loop_controller.update_status(passed=False)
            iteration_data = {
                "iteration": loop_controller.current_iteration,
                "generator_output": None,
                "metrics": {},
                "feedback": f"生成失败: {generator_result.get('error')}",
                "passed": False,
                "timestamp": datetime.now().isoformat()
            }
            loop_controller.record_iteration(iteration_data)
        else:
            # ========== Agent B: 评价 ==========
            print("\n[Agent B] 执行评价...")

            metrics = evaluator.evaluate(
                video_path=str(output_path),
                audio_path=str(audio_path)
            )

            print(f"[Agent B] 指标结果: {metrics.get('metrics', {})}")

            # 检查是否触发人工评判
            should_human_review = (
                not args.skip_human_review and
                evaluator.should_trigger_human_review(metrics.get("metrics", {}))
            )

            human_review_result = None
            if should_human_review:
                print("\n[Agent B] AI指标达标，生成人工评判样本...")
                review_sample = evaluator.format_human_review_sample(
                    video_path=str(output_path),
                    audio_path=str(audio_path),
                    iteration=loop_controller.current_iteration
                )

                # TODO: 后续集成 Gradio 人工评判界面
                print("[Agent B] 人工评判待集成，当前跳过...")
                print(f"[Agent B] 评判样本已保存到: {review_sample.get('summary_video_path')}")

            # 生成反馈
            feedback = evaluator.generate_feedback(
                metrics=metrics.get("metrics", {}),
                human_review_result=human_review_result
            )

            # 判断是否通过
            passed = metrics.get("passed", False)
            if human_review_result:
                # TODO: 结合人工评判结果判断
                pass

            loop_controller.update_status(passed=passed)

            # 记录迭代
            iteration_data = {
                "iteration": loop_controller.current_iteration,
                "generator_output": str(output_path),
                "metrics": metrics.get("metrics", {}),
                "human_review": human_review_result,
                "feedback": feedback,
                "passed": passed,
                "timestamp": datetime.now().isoformat()
            }
            loop_controller.record_iteration(iteration_data)

            # 更新检查点
            import shutil
            shutil.copy(output_path, checkpoint_path)

            print(f"\n[Agent B] 评价结果: {'通过' if passed else '未通过'}")
            print(f"[Agent B] 反馈:\n{feedback}")

        # 检查收敛
        stop_decision = loop_controller.should_stop()

        print(f"\n[Loop Controller] 状态: {stop_decision}")

        if stop_decision["should_stop"]:
            print("\n" + "=" * 60)
            print("迭代循环结束")
            print(f"原因: {stop_decision['reason']}")
            print("=" * 60)

            if stop_decision["status"] == "success":
                best = loop_controller.get_best_iteration()
                if best:
                    print(f"\n最佳结果: {best.get('generator_output')}")
                    # 复制最终结果
                    import shutil
                    final_output = Path(args.output)
                    shutil.copy(best.get("generator_output"), final_output)
                    print(f"最终视频已保存到: {final_output}")
            break

    print("\n[完成] 迭代历史已保存到:", workspace_dir / "iteration_history.json")


if __name__ == "__main__":
    main()
