"""Dataset configuration."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""

    name: str
    source: str
    text_field: str
    splits: dict[str, str]

    def __post_init__(self):
        """Validate config after initialization."""
        if "train" not in self.splits:
            raise ValueError(
                "Dataset config must contain 'train' key in splits dictionary"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetConfig":
        """
        Load config from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            DatasetConfig instance
        """
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)
