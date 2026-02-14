"""Llama 3-style transformer block.

A pre-norm decoder block built with:
1) RMSNorm + (MHA/GQA/MQA attention) + residual
2) RMSNorm + SwiGLU FFN + residual

RoPE frequencies are passed from the caller (model-level), following the same
pattern used in Mini-LLM style implementations.
"""

import torch
import torch.nn as nn

from tiny_lm.model.feedforward import SwiGLU
from tiny_lm.model.normalization import RMSNorm
from tiny_lm.model.utils import create_attention


class Llama3Block(nn.Module):
    """Single Llama 3-style decoder block.

    Args:
        d_model: Model dimension
        n_heads: Number of query heads
        context_length: Maximum sequence length
        n_kv_heads: Number of key/value heads (None -> n_heads)
        ffn_hidden_dim: SwiGLU hidden dimension (None -> 4 * d_model)
        multiple_of: Round FFN hidden dim up to this multiple
        norm_eps: Epsilon for RMSNorm
        attn_dropout: Dropout probability on attention weights
        resid_dropout: Dropout on residual branches
        ffn_dropout: Dropout inside SwiGLU gated representation
        qkv_bias: Whether to use bias in attention projections
        ffn_bias: Whether SwiGLU linear layers use bias
        attn_backend: Attention backend ("manual" or "sdp")
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_length: int,
        n_kv_heads: int | None = None,
        ffn_hidden_dim: int | None = None,
        multiple_of: int = 1,
        norm_eps: float = 1e-6,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        qkv_bias: bool = False,
        ffn_bias: bool = False,
        attn_backend: str = "manual",
    ):
        super().__init__()

        self.attn = create_attention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            context_length=context_length,
            dropout=attn_dropout,
            qkv_bias=qkv_bias,
            backend=attn_backend,
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            hidden_dim=ffn_hidden_dim,
            multiple_of=multiple_of,
            dropout=ffn_dropout,
            bias=ffn_bias,
        )

        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)

        self.dropout1 = nn.Dropout(resid_dropout)
        self.dropout2 = nn.Dropout(resid_dropout)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply the block with optional RoPE frequencies for attention."""
        h_attn = self.attn(self.attn_norm(x), freqs_cis=freqs_cis)
        h = x + self.dropout1(h_attn)

        h_ffn = self.ffn(self.ffn_norm(h))
        return h + self.dropout2(h_ffn)
