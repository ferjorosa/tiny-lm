"""Configuration for binary token stream data."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class BinDataConfig:
    """Configuration for BinTokenDataModule."""

    train_path: str
    val_path: str
    block_size: int
    stride: int
    dtype: str
    eos_token_id: int | None
    batch_size: int
    num_workers: int
    pin_memory: bool
    drop_last: bool

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.stride > self.block_size:
            raise ValueError("stride cannot be larger than block_size")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BinDataConfig":
        """
        Load config from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            BinDataConfig instance
        """
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)
