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
    warmup_steps: int
    max_steps: int
    min_lr: float
    save_every_n_steps: int
    save_top_k: int

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.accumulate_grad_batches <= 0:
            raise ValueError("accumulate_grad_batches must be positive")
        if self.grad_clip_norm < 0:
            raise ValueError("grad_clip_norm must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.min_lr < 0:
            raise ValueError("min_lr must be non-negative")
        if self.scheduler not in {"cosine", "none"}:
            raise ValueError("scheduler must be 'cosine' or 'none'")

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
