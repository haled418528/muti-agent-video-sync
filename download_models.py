"""
下载预训练模型脚本
使用 Wav2Lip 官方预训练模型
"""

import os
import sys
from pathlib import Path
import urllib.request

# Wav2Lip 官方模型下载链接
MODEL_URLS = {
    "wav2lip_gan": {
        "url": "https://github.com/Rudrabha/Wav2Lip/releases/download/models/wav2lip_gan.pth",
        "description": "Wav2Lip GAN 模型（推荐，用于推理）"
    },
    "wav2lip": {
        "url": "https://github.com/Rudrabha/Wav2Lip/releases/download/models/wav2lip.pth",
        "description": "Wav2Lip 基础模型"
    },
    "s3fd": {
        "url": "https://github.com/Rudrabha/Wav2Lip/releases/download/models/s3fd.pth",
        "description": "S3FD 人脸检测模型"
    },
    "syncnet": {
        "url": "https://github.com/Rudrabha/Wav2Lip/releases/download/models/syncnet.pth",
        "description": "SyncNet 同步评估模型"
    },
    # Wav2Lip-HQ (更高质量版本)
    "wav2lip_hq": {
        "url": "https://github.com/Rudrabha/Wav2Lip/releases/download/models/wav2lip_hq.pth",
        "description": "Wav2Lip 高质量版本"
    }
}

CHECKPOINTS_DIR = Path("checkpoints")


def download_file(url: str, destination: Path, description: str = "") -> bool:
    """
    下载文件

    Args:
        url: 下载链接
        destination: 保存路径
        description: 模型描述

    Returns:
        是否成功
    """
    try:
        print(f"\n正在下载: {description}")
        print(f"链接: {url}")
        print(f"保存到: {destination}")

        # 创建进度回调
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
            sys.stdout.write(f"\r  进度: {percent:.1f}%")
            sys.stdout.flush()

        # 下载文件
        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print(f"\n  ✓ 下载完成! 大小: {destination.stat().st_size / 1024 / 1024:.2f} MB")
        return True

    except Exception as e:
        print(f"\n  ✗ 下载失败: {e}")
        if destination.exists():
            destination.unlink()
        return False


def download_all_models():
    """下载所有预训练模型"""
    # 创建目录
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Multi-Agent Video Lip-Sync - 模型下载")
    print("=" * 60)
    print(f"\n模型保存目录: {CHECKPOINTS_DIR.absolute()}")
    print(f"可用模型: {len(MODEL_URLS)}")
    print()

    success_count = 0
    fail_count = 0

    for model_name, info in MODEL_URLS.items():
        print(f"\n[{model_name}]")
        dest_path = CHECKPOINTS_DIR / f"{model_name}.pth"

        if dest_path.exists():
            size_mb = dest_path.stat().st_size / 1024 / 1024
            print(f"  ✓ 已存在，跳过 ({size_mb:.2f} MB): {dest_path}")
            success_count += 1
            continue

        success = download_file(info["url"], dest_path, info["description"])
        if success:
            success_count += 1
        else:
            print(f"  ⚠ 下载 {model_name} 失败，请稍后重试或手动下载")
            fail_count += 1

    print("\n" + "=" * 60)
    print("下载完成!")
    print(f"  成功: {success_count}/{len(MODEL_URLS)}")
    if fail_count > 0:
        print(f"  失败: {fail_count}/{len(MODEL_URLS)}")
        print("  提示: 失败的文件可以稍后单独下载")
    print("=" * 60)

    return fail_count == 0


def download_single_model(model_name: str):
    """下载单个模型"""
    if model_name not in MODEL_URLS:
        print(f"未知模型: {model_name}")
        print(f"\n可用模型:")
        for name, info in MODEL_URLS.items():
            print(f"  - {name}: {info['description']}")
        return False

    info = MODEL_URLS[model_name]
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    dest_path = CHECKPOINTS_DIR / f"{model_name}.pth"
    if dest_path.exists():
        size_mb = dest_path.stat().st_size / 1024 / 1024
        print(f"模型已存在 ({size_mb:.2f} MB): {dest_path}")
        return True

    return download_file(info["url"], dest_path, info["description"])


def list_models():
    """列出所有可用模型"""
    print("\n可用模型:")
    print("-" * 40)
    for model_name, info in MODEL_URLS.items():
        dest_path = CHECKPOINTS_DIR / f"{model_name}.pth"
        status = "✓ 已下载" if dest_path.exists() else "✗ 未下载"
        print(f"  {model_name}")
        print(f"    {info['description']}")
        print(f"    {status}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            list_models()
        else:
            # 下载指定模型
            for arg in sys.argv[1:]:
                download_single_model(arg)
    else:
        # 交互模式
        print("\n请选择操作:")
        print("  1. 下载所有模型")
        print("  2. 下载指定模型")
        print("  3. 查看已下载模型")
        print("  4. 退出")

        choice = input("\n请输入选项 (1-4): ").strip()

        if choice == "1":
            download_all_models()
        elif choice == "2":
            list_models()
            model_name = input("\n请输入模型名称: ").strip()
            download_single_model(model_name)
        elif choice == "3":
            list_models()
        else:
            print("退出")
