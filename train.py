"""训练 MNIST 分类模型。"""

import argparse

import torch
from torch import nn

from config import DEFAULT_CONFIG
from data import create_dataloaders
from model import MNISTNet
from utils import (
    ensure_output_dirs,
    get_device,
    plot_history,
    save_checkpoint,
    save_history,
    set_seed,
)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """训练一轮并返回样本平均损失。"""

    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """计算分类准确率，返回 0 到 100 之间的百分数。"""

    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 MNIST 手写数字分类模型")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG.learning_rate)
    parser.add_argument(
        "--download",
        action="store_true",
        help="数据不存在时允许下载 MNIST；不传此参数则绝不下载",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG
    set_seed(config.seed)
    ensure_output_dirs(config.checkpoint_dir, config.figure_dir)
    device = get_device()
    print(f"使用设备：{device}")

    try:
        train_loader, test_loader = create_dataloaders(
            config.data_dir,
            batch_size=args.batch_size,
            num_workers=config.num_workers,
            download=args.download,
        )
    except RuntimeError as error:
        raise SystemExit(
            "MNIST 数据不存在。首次运行请使用：python train.py --download"
        ) from error

    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history: dict[str, list[float]] = {"train_loss": [], "test_accuracy": []}
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        accuracy = evaluate_accuracy(model, test_loader, device)
        history["train_loss"].append(loss)
        history["test_accuracy"].append(accuracy)
        print(f"Epoch {epoch:02d}/{args.epochs} | loss={loss:.4f} | accuracy={accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_checkpoint(config.best_model_path, model, optimizer, epoch, accuracy)

    save_history(history, config.output_dir / "history.json")
    plot_history(history, config.figure_dir / "training_history.png")
    print(f"训练完成，最佳准确率：{best_accuracy:.2f}%")
    print(f"最佳模型：{config.best_model_path}")


if __name__ == "__main__":
    main()

