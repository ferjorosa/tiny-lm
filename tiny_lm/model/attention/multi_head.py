"""Multi-Head Attention for GPT-2.

Multi-head attention allows the model to jointly attend to information from
different representation subspaces at different positions.

Paper: "Attention is All You Need" (Vaswani et al., 2017)
https://arxiv.org/abs/1706.03762
"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with causal masking.

    Splits the input into multiple heads, computes scaled dot-product attention
    for each head independently, then concatenates and projects the results.

    Args:
        d_model: Dimension of model embeddings
        n_heads: Number of attention heads
        context_length: Maximum sequence length (for causal mask)
        dropout: Dropout probability for attention weights
        qkv_bias: Whether to use bias in Q, K, V projections
    """

    causal_mask: torch.Tensor

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_length: int,
        dropout: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_k = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_v = nn.Linear(d_model, d_model, bias=qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Causal mask (upper triangular matrix of ones)
        # This prevents attending to future tokens
        causal_mask = torch.triu(
            torch.ones(context_length, context_length), diagonal=1
        ).bool()
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V: (batch_size, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape and split into multiple heads
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose to (batch_size, n_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores: Q @ K^T
        # (batch_size, n_heads, seq_len, head_dim) @ (batch_size, n_heads, head_dim, seq_len)
        # -> (batch_size, n_heads, seq_len, seq_len)
        attn_scores = Q @ K.transpose(-2, -1)

        # Scale by sqrt(head_dim) for stability
        attn_scores = attn_scores / (self.head_dim**0.5)

        # Apply causal mask (prevent attending to future tokens)
        # Truncate mask to current sequence length
        mask = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, head_dim)
        # -> (batch_size, n_heads, seq_len, head_dim)
        context = attn_weights @ V

        # Transpose back: (batch_size, n_heads, seq_len, head_dim)
        # -> (batch_size, seq_len, n_heads, head_dim)
        context = context.transpose(1, 2)

        # Concatenate heads: (batch_size, seq_len, n_heads, head_dim)
        # -> (batch_size, seq_len, d_model)
        context = context.contiguous().view(batch_size, seq_len, self.d_model)

        # Final output projection
        output = self.out_proj(context)

        return output
