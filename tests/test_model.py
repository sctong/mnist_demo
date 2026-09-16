"""模型结构的基础测试。"""

import torch

from model import MNISTNet


def test_model_output_shape() -> None:
    model = MNISTNet()
    images = torch.randn(4, 1, 28, 28)
    output = model(images)
    assert output.shape == (4, 10)


def test_model_has_trainable_parameters() -> None:
    model = MNISTNet()
    assert sum(parameter.numel() for parameter in model.parameters()) > 0

