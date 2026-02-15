"""Attention using PyTorch's scaled_dot_product_attention.

PyTorch's SDPA provides efficient kernel fusion and memory-optimized attention
computation. It handles GQA/MQA natively without explicit head repetition.

Paper: "Self-attention Does Not Need O(n²) Memory" (Rabe & Staats, 2021)
https://arxiv.org/abs/2112.05682
"""

import torch
import torch.nn as nn

from tiny_lm.model.position import apply_rotary_emb


class SDPAttention(nn.Module):
    """Causal self-attention using PyTorch's scaled_dot_product_attention.

    Supports MHA, GQA, and MQA via n_kv_heads. Uses PyTorch's SDPA for
    efficient kernel fusion and memory-optimized attention computation.

    Args:
        d_model: Dimension of model embeddings
        n_heads: Number of attention heads
        n_kv_heads: Number of key/value heads.
            - If None, defaults to n_heads (standard MHA)
            - If 1, uses MQA
            - If between 1 and n_heads, uses GQA
        dropout: Dropout probability for attention weights
        qkv_bias: Whether to use bias in Q, K, V projections
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if n_kv_heads is None:
            n_kv_heads = n_heads
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads

        # Linear projections: Q uses n_heads, K/V use n_kv_heads for GQA/MQA
        self.W_q = nn.Linear(d_model, n_heads * self.head_dim, bias=qkv_bias)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=qkv_bias)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout applied during training
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply causal self-attention using PyTorch SDPA.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            freqs_cis: Optional RoPE frequencies of shape (seq_len, head_dim // 2).
                If provided, applies rotary embedding to Q and K.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape into heads: (batch, seq_len, n_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE if frequencies provided
        if freqs_cis is not None:
            Q, K = apply_rotary_emb(Q, K, freqs_cis)

        # Transpose to (batch, n_heads, seq_len, head_dim) for Q
        # K, V keep n_kv_heads for native GQA/MQA support
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # PyTorch SDPA handles GQA/MQA natively without explicit repeat
        # is_causal=True applies causal masking efficiently
        # enable_gqa=True allows native GQA support when n_kv_heads < n_heads
        context = torch.nn.functional.scaled_dot_product_attention(
            Q,
            K,
            V,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
            enable_gqa=True,
        )

        # Transpose back: (batch, n_heads, seq_len, head_dim) -> (batch, seq_len, n_heads, head_dim)
        context = context.transpose(1, 2)

        # Concatenate heads and project
        context = context.contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(context)
