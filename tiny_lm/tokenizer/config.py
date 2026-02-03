"""Tokenizer configuration."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer training and usage."""

    name: str
    type: str
    vocab_size: int
    dataset_config: str
    special_tokens: dict[str, str]
    output_dir: str
    tokenized_output_dir: str
    val_split: float = 0.05

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TokenizerConfig":
        """
        Load config from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            TokenizerConfig instance
        """
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)
