"""MNIST 数据预处理与 DataLoader 创建。"""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# MNIST 官方训练集的均值和标准差，用于标准化输入。
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def get_transform() -> transforms.Compose:
    """返回训练和测试共同使用的图像预处理。"""

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    download: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """创建训练集和测试集加载器。

    默认不下载数据。只有调用者明确传入 ``download=True`` 时，
    torchvision 才会尝试下载 MNIST。
    """

    root = Path(data_dir)
    transform = get_transform()
    train_dataset = datasets.MNIST(
        root=root, train=True, transform=transform, download=download
    )
    test_dataset = datasets.MNIST(
        root=root, train=False, transform=transform, download=download
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, test_loader

