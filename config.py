"""项目的默认配置。"""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    """集中管理常用超参数，便于初学者查看和修改。"""

    data_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "outputs"
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 5
    seed: int = 42
    num_workers: int = 0  # Windows 初学环境使用 0 更稳定

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def figure_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def best_model_path(self) -> Path:
        return self.checkpoint_dir / "best_model.pth"


DEFAULT_CONFIG = Config()

