"""Model configuration classes."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class GPT2Config:
    """Configuration for GPT-2 model."""

    vocab_size: int = 8000
    context_length: int = 1024
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072  # 4 * d_model
    dropout: float = 0.1
    emb_dropout: float = 0.1
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1

    def __post_init__(self):
        """Validate configuration."""
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GPT2Config":
        """
        Load config from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            GPT2Config instance
        """
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)


@dataclass
class Llama3Config:
    """Configuration for Llama 3-style decoder model."""

    vocab_size: int = 8192
    context_length: int = 1024
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_hidden_dim: int | None = None
    multiple_of: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    emb_dropout: float = 0.0
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    ffn_dropout: float = 0.0
    qkv_bias: bool = False
    ffn_bias: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Llama3Config":
        """Load config from YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)
