"""
下载预训练模型脚本
TODO: 实现模型下载逻辑
"""

import os
import sys
from pathlib import Path
import urllib.request
import zipfile

# 模型下载链接（TODO: 确认实际链接）
MODEL_URLS = {
    "wav2lip_gan": "https://github.com/Rudrabha/Wav2Lip/releases/download/models/wav2lip_gan.pth",
    "wav2lip": "https://github.com/Rudrabha/Wav2Lip/releases/download/models/wav2lip.pth",
    "s3fd": "https://github.com/Rudrabha/Wav2Lip/releases/download/models/s3fd.pth",
    "syncnet": "https://github.com/Rudrabha/Wav2Lip/releases/download/models/syncnet.pth",
}

CHECKPOINTS_DIR = Path("checkpoints")


def download_file(url: str, destination: Path) -> bool:
    """
    下载文件

    Args:
        url: 下载链接
        destination: 保存路径

    Returns:
        是否成功
    """
    try:
        print(f"正在下载: {url}")
        print(f"保存到: {destination}")

        # 创建进度回调
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
            sys.stdout.write(f"\r进度: {percent:.1f}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print("\n下载完成!")
        return True

    except Exception as e:
        print(f"\n下载失败: {e}")
        return False


def download_all_models():
    """下载所有预训练模型"""
    # 创建目录
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Multi-Agent Video Lip-Sync 模型下载")
    print("=" * 60)
    print(f"模型保存目录: {CHECKPOINTS_DIR.absolute()}")
    print()

    for model_name, url in MODEL_URLS.items():
        print(f"\n[{model_name}]")
        dest_path = CHECKPOINTS_DIR / f"{model_name}.pth"

        if dest_path.exists():
            print(f"  已存在，跳过: {dest_path}")
            continue

        success = download_file(url, dest_path)
        if not success:
            print(f"  下载 {model_name} 失败，请手动下载后放置到 checkpoints 目录")


def download_single_model(model_name: str):
    """下载单个模型"""
    if model_name not in MODEL_URLS:
        print(f"未知模型: {model_name}")
        print(f"可用模型: {list(MODEL_URLS.keys())}")
        return

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    dest_path = CHECKPOINTS_DIR / f"{model_name}.pth"
    if dest_path.exists():
        print(f"模型已存在: {dest_path}")
        return

    url = MODEL_URLS[model_name]
    download_file(url, dest_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 下载指定模型
        for arg in sys.argv[1:]:
            download_single_model(arg)
    else:
        # 下载所有模型
        download_all_models()

    print("\n模型下载完成!")
