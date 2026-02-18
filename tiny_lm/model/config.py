"""Model configuration classes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


def _load_yaml_dict(path: str | Path) -> dict[str, object]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


@dataclass
class GPT2Config:
    """Configuration for GPT-2 model."""

    model_type: Literal["gpt2"]
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
    attn_backend: str = "manual"  # "manual" or "sdp"

    def __post_init__(self):
        """Validate configuration."""
        if self.model_type != "gpt2":
            raise ValueError(f"model_type must be 'gpt2', got {self.model_type}")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.attn_backend not in ("manual", "sdp"):
            raise ValueError(
                f"attn_backend must be 'manual' or 'sdp', got {self.attn_backend}"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GPT2Config":
        """
        Load config from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            GPT2Config instance
        """
        return cls(**_load_yaml_dict(path))


@dataclass
class Llama3Config:
    """Configuration for Llama 3-style decoder model."""

    model_type: Literal["llama3"]
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
    attn_backend: str = "manual"  # "manual" or "sdp"

    def __post_init__(self):
        """Validate configuration."""
        if self.model_type != "llama3":
            raise ValueError(f"model_type must be 'llama3', got {self.model_type}")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.attn_backend not in ("manual", "sdp"):
            raise ValueError(
                f"attn_backend must be 'manual' or 'sdp', got {self.attn_backend}"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Llama3Config":
        """Load config from YAML file."""
        return cls(**_load_yaml_dict(path))
