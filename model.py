"""用于 MNIST 分类的小型卷积神经网络。"""

import torch
from torch import nn


class MNISTNet(nn.Module):
    """输入形状为 [批大小, 1, 28, 28]，输出 10 类 logits。"""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # 28×28 -> 28×28 -> 14×14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 14×14 -> 14×14 -> 7×7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features)

