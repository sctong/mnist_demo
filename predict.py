"""使用已训练模型预测一张外部手写数字图片。"""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from config import DEFAULT_CONFIG
from data import MNIST_MEAN, MNIST_STD
from model import MNISTNet
from utils import get_device, load_model_weights


def prepare_image(image_path: Path, dark_on_light: bool = True) -> torch.Tensor:
    """将外部图片转换为接近 MNIST 的 1×28×28 张量。"""

    image = Image.open(image_path).convert("L")
    if dark_on_light:
        # MNIST 是黑底白字，常见纸面照片则是白底黑字。
        image = ImageOps.invert(image)

    bbox = image.getbbox()
    if bbox is None:
        raise ValueError("图片中没有检测到可见内容。")
    image = image.crop(bbox)

    # 保持宽高比，将数字缩放到 20×20 范围，再居中放入 28×28 画布。
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), color=0)
    offset = ((28 - image.width) // 2, (28 - image.height) // 2)
    canvas.paste(image, offset)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
    )
    return transform(canvas).unsqueeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预测单张手写数字图片")
    parser.add_argument("image", type=Path, help="待预测图片路径")
    parser.add_argument("--model", type=Path, default=DEFAULT_CONFIG.best_model_path)
    parser.add_argument(
        "--white-on-black",
        action="store_true",
        help="图片已经是黑底白字时使用",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = get_device()
    model = MNISTNet().to(device)
    load_model_weights(model, args.model, device)
    model.eval()

    image = prepare_image(args.image, dark_on_light=not args.white_on_black).to(device)
    probabilities = torch.softmax(model(image), dim=1).squeeze(0)
    predicted = probabilities.argmax().item()
    confidence = probabilities[predicted].item() * 100
    print(f"预测数字：{predicted}")
    print(f"置信度：{confidence:.2f}%")


if __name__ == "__main__":
    main()

