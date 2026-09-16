"""训练、评估和绘图共用的辅助函数。"""

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    """固定随机种子，让多次运行的结果尽量可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """优先选择 CUDA，否则使用 CPU。"""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_output_dirs(checkpoint_dir: Path, figure_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    accuracy: float,
) -> None:
    """保存恢复训练所需的信息，而不仅是模型参数。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "accuracy": accuracy,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_model_weights(model: nn.Module, path: Path, device: torch.device) -> dict[str, Any]:
    """加载检查点并把模型权重写入给定模型。"""

    if not path.exists():
        raise FileNotFoundError(f"未找到模型文件：{path}。请先运行训练。")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def save_history(history: dict[str, list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_history(history: dict[str, list[float]], path: Path) -> None:
    """将每轮损失与准确率保存为图片。"""

    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, history["test_accuracy"], marker="o")
    axes[1].set(title="Test accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)

