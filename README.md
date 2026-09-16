# MNIST 手写数字分类入门项目

这是一个面向 PyTorch 初学者的完整图像分类项目。它使用小型卷积神经网络（CNN）识别 0～9 的手写数字，并将数据处理、模型定义、训练、评估和单图预测拆分到不同文件中。

## 1. 学习目标

完成本项目后，你可以理解：

- 图像如何被转换为神经网络能够处理的张量；
- 卷积层、池化层和全连接层的基本作用；
- 损失计算、反向传播和参数更新的训练流程；
- 训练模式与评估模式的区别；
- 如何保存模型并用它预测新图片。

## 2. 项目结构

```text
mnist_demo/
├── README.md              # 项目说明和运行指南
├── requirements.txt       # Python 第三方依赖
├── config.py              # 默认超参数和路径
├── data.py                # 数据预处理与 DataLoader
├── model.py               # 小型卷积神经网络
├── train.py               # 模型训练入口
├── evaluate.py            # 测试集评估入口
├── predict.py             # 单张图片预测入口
├── utils.py               # 随机种子、设备、模型存取和绘图
├── tests/                 # 不下载数据的基础测试
├── data/                  # MNIST 数据，首次训练时生成
└── outputs/               # 模型、曲线和评估图片，运行后生成
```

## 3. 环境准备

推荐使用 Python 3.10 或更高版本，并创建独立虚拟环境。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

> 本仓库不会在导入模块时自动安装依赖或下载数据。以上安装命令需要你主动执行。

## 4. 首次训练

首次运行时，本地还没有 MNIST，因此必须显式添加 `--download`：

```powershell
python train.py --download
```

下载完成后，再次训练不需要网络，也不需要 `--download`：

```powershell
python train.py
```

可以通过命令行修改几个常用参数：

```powershell
python train.py --epochs 10 --batch-size 128 --learning-rate 0.001
```

训练产物包括：

- `outputs/checkpoints/best_model.pth`：测试准确率最高的模型；
- `outputs/history.json`：每轮损失和准确率；
- `outputs/figures/training_history.png`：训练曲线。

## 5. 评估模型

```powershell
python evaluate.py
```

程序会输出测试集准确率，并生成 `outputs/figures/evaluation_samples.png`。如果只下载过部分数据，也可以显式运行 `python evaluate.py --download`。

## 6. 预测自己的图片

对于常见的白底黑字图片：

```powershell
python predict.py path\to\digit.png
```

如果图片本身已经是黑底白字：

```powershell
python predict.py path\to\digit.png --white-on-black
```

建议图片中只包含一个数字，背景干净，数字尽量位于中央。外部图片与 MNIST 的书写风格差异很大时，预测可能不准确。

## 7. 运行测试

```powershell
python -m pytest -q
```

这些测试只使用内存中创建的假数据，不会下载 MNIST。

## 8. 核心训练过程

每一个批次都会执行以下步骤：

1. 将图片和标签移动到 CPU 或 GPU；
2. 前向传播，得到 10 个类别的原始分数；
3. 使用交叉熵计算预测误差；
4. 清空上一批次的梯度；
5. 反向传播计算梯度；
6. Adam 优化器更新模型参数。

默认训练 5 轮。项目会自动使用 CUDA GPU（如果 PyTorch 检测到可用 CUDA），否则使用 CPU。
Git practice: first update after initial push.

