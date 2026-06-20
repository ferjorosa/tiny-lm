"""SwiGLU feed-forward network.

SwiGLU uses two parallel projections:
- gate projection with SiLU activation
- value projection
Then multiplies them elementwise and projects back to model dimension.

Paper: "GLU Variants Improve Transformer" (Shazeer, 2020)
https://arxiv.org/abs/2002.05202
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    Architecture: w2(SiLU(w1(x)) * w3(x))

    Args:
        d_model: Input and output dimension
        hidden_dim: Intermediate dimension. If None, defaults to 4 * d_model
        multiple_of: Round hidden_dim up to this multiple
        dropout: Dropout after gated activation
        bias: Whether linear layers use bias
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        multiple_of: int = 1,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * d_model

        if multiple_of > 1:
            hidden_dim = ((hidden_dim + multiple_of - 1) // multiple_of) * multiple_of

        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=bias)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        gated = F.silu(self.w1(x)) * self.w3(x)
        gated = self.dropout(gated)
        return self.w2(gated)
