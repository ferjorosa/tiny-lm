"""Model configuration classes."""

from dataclasses import dataclass


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
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
