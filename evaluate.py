"""在 MNIST 测试集上评估已训练模型。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from config import DEFAULT_CONFIG
from data import MNIST_MEAN, MNIST_STD, create_dataloaders
from model import MNISTNet
from utils import get_device, load_model_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 MNIST 模型")
    parser.add_argument("--model", type=str, default=str(DEFAULT_CONFIG.best_model_path))
    parser.add_argument(
        "--download",
        action="store_true",
        help="数据不存在时允许下载 MNIST；不传此参数则绝不下载",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = get_device()
    try:
        _, test_loader = create_dataloaders(
            DEFAULT_CONFIG.data_dir,
            batch_size=DEFAULT_CONFIG.batch_size,
            num_workers=DEFAULT_CONFIG.num_workers,
            download=args.download,
        )
    except RuntimeError as error:
        raise SystemExit("MNIST 数据不存在，请先运行训练命令并添加 --download。") from error

    model = MNISTNet().to(device)
    checkpoint = load_model_weights(model, Path(args.model), device)
    model.eval()

    correct = 0
    total = 0
    sample_images = None
    sample_labels = None
    sample_predictions = None
    for images, labels in test_loader:
        predictions = model(images.to(device)).argmax(dim=1).cpu()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        if sample_images is None:
            sample_images = images[:10]
            sample_labels = labels[:10]
            sample_predictions = predictions[:10]

    print(f"检查点轮次：{checkpoint.get('epoch', '未知')}")
    print(f"测试集准确率：{100.0 * correct / total:.2f}% ({correct}/{total})")

    # 保存前 10 个样本的预测结果，方便直观检查。
    assert sample_images is not None and sample_labels is not None and sample_predictions is not None
    figure, axes = plt.subplots(2, 5, figsize=(10, 4))
    for axis, image, label, prediction in zip(
        axes.flat, sample_images, sample_labels, sample_predictions
    ):
        restored = image.squeeze().numpy() * MNIST_STD[0] + MNIST_MEAN[0]
        axis.imshow(restored, cmap="gray")
        axis.set_title(f"pred={prediction.item()}, true={label.item()}")
        axis.axis("off")
    figure.tight_layout()
    output_path = DEFAULT_CONFIG.figure_dir / "evaluation_samples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"样本预测图：{output_path}")


if __name__ == "__main__":
    main()
