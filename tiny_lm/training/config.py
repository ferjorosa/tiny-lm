"""Training configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import yaml


Precision: TypeAlias = Literal[
    64,
    32,
    16,
    "bf16-mixed",
    "16-mixed",
    "32-true",
    "bf16-true",
]


@dataclass
class TrainingConfig:
    """Configuration for training components."""

    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    ignore_index: int
    precision: Precision
    accumulate_grad_batches: int
    grad_clip_norm: float
    scheduler: str
    warmup_ratio: float
    max_steps: int
    min_lr: float
    save_every_n_steps: int
    val_every_n_steps: int
    system_metrics_every_n_steps: int
    batch_size: int
    run_name: str | None = None
    resume_from_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.accumulate_grad_batches <= 0:
            raise ValueError("accumulate_grad_batches must be positive")
        if self.grad_clip_norm < 0:
            raise ValueError("grad_clip_norm must be non-negative")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be between 0 and 1")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.min_lr < 0:
            raise ValueError("min_lr must be non-negative")
        if self.scheduler not in {"cosine", "none"}:
            raise ValueError("scheduler must be 'cosine' or 'none'")
        if self.val_every_n_steps <= 0:
            raise ValueError("val_every_n_steps must be positive")
        if self.system_metrics_every_n_steps <= 0:
            raise ValueError("system_metrics_every_n_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """
        Load config from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            TrainingConfig instance
        """
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)
