"""Rotary positional embeddings (RoPE).

RoPE encodes position by rotating query/key vectors in attention head space.
Unlike additive position embeddings, rotations preserve vector norms and inject
relative-position information directly into attention logits.

Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Precompute complex rotary frequencies for a maximum sequence length.

    Args:
        dim: Attention head dimension (must be even)
        max_seq_len: Maximum sequence length to precompute
        theta: Base frequency constant (default: 10000.0)
    """

    _freqs_cis: torch.Tensor

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")

        self.dim = dim
        self.max_seq_len = max_seq_len

        # 1) Rotation frequencies for each 2D pair in head_dim.
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        # 2) Position index: [0, 1, 2, ..., max_seq_len - 1]
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        # 3) Angle matrix: (max_seq_len, dim // 2)
        freqs = torch.outer(positions, inv_freq)
        # 4) Convert angles to complex unit vectors: cos + i*sin
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("_freqs_cis", freqs_cis)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Return rotary frequencies for the current sequence length.

        Args:
            seq_len: Current sequence length

        Returns:
            Complex tensor of shape (seq_len, dim // 2)
        """
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum sequence length "
                f"{self.max_seq_len}"
            )
        return self._freqs_cis[:seq_len]


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embedding to query and key tensors.

    Expected shapes:
        xq: (batch, seq_len, n_heads, head_dim)
        xk: (batch, seq_len, n_kv_heads, head_dim)
        freqs_cis: (seq_len, head_dim // 2), complex

    Returns:
        Rotated (xq, xk) with same shape and dtype as inputs.
    """
    if xq.shape[1] != xk.shape[1] or xq.shape[-1] != xk.shape[-1]:
        raise ValueError("xq and xk must share seq_len and head_dim")
    if xq.shape[-1] % 2 != 0:
        raise ValueError("head_dim must be even to apply RoPE")

    seq_len = xq.shape[1]
    freqs_cis = freqs_cis[:seq_len]

    # 1) Group real values into pairs: (..., head_dim) -> (..., head_dim // 2)
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 2) Broadcast freqs over batch and head axes:
    #    (seq_len, head_dim // 2) -> (1, seq_len, 1, head_dim // 2)
    freqs_shape = [1] * xq_complex.ndim
    freqs_shape[1] = xq_complex.shape[1]
    freqs_shape[-1] = xq_complex.shape[-1]
    freqs_cis = freqs_cis.view(*freqs_shape)

    # 3) Complex multiply applies the rotation; convert back to real layout.
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)
