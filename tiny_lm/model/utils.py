"""Model-level utilities."""

import torch.nn as nn

from tiny_lm.model.attention import MultiHeadAttention, SDPAttention


def create_attention(
    d_model: int,
    n_heads: int,
    n_kv_heads: int | None,
    context_length: int,
    dropout: float,
    qkv_bias: bool,
    backend: str,
) -> nn.Module:
    """Create attention module based on backend.

    Args:
        d_model: Model dimension
        n_heads: Number of query heads
        n_kv_heads: Number of key/value heads
        context_length: Maximum sequence length (used by manual backend)
        dropout: Dropout probability
        qkv_bias: Whether to use bias in projections
        backend: "manual" or "sdp"

    Returns:
        Attention module
    """
    if backend == "sdp":
        return SDPAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )
    return MultiHeadAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        context_length=context_length,
        dropout=dropout,
        qkv_bias=qkv_bias,
    )
