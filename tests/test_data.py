"""数据预处理的基础测试，不会下载 MNIST。"""

from PIL import Image

from data import get_transform


def test_transform_output_shape() -> None:
    image = Image.new("L", (28, 28), color=0)
    tensor = get_transform()(image)
    assert tensor.shape == (1, 28, 28)

