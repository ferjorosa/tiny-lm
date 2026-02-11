"""GPT-2 Transformer Block.

A single transformer block using pre-norm architecture. Each block consists of:
1. Multi-head self-attention with residual connection
2. Feed-forward network with residual connection

Pre-norm (normalize before sublayer) is more stable than post-norm for deep networks.

Paper: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
"""

import torch
import torch.nn as nn
from tiny_lm.model.attention import MultiHeadAttention
from tiny_lm.model.feedforward import FeedForward
from tiny_lm.model.normalization import LayerNorm


class GPT2Block(nn.Module):
    """GPT-2 Transformer Block with pre-norm architecture.

    Architecture:
        x -> LayerNorm -> MultiHeadAttention -> Residual ->
        x -> LayerNorm -> FeedForward -> Residual -> output

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        context_length: Maximum sequence length
        activation: Activation function for FFN
        attn_dropout: Dropout for attention weights
        resid_dropout: Dropout for residual connections
        ffn_dropout: Dropout in feed-forward network
        qkv_bias: Whether to use bias in attention QKV projections
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        context_length: int,
        activation: nn.Module,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()

        # Multi-head attention
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            context_length=context_length,
            dropout=attn_dropout,
            qkv_bias=qkv_bias,
        )

        # Feed-forward network
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout=ffn_dropout,
        )

        # Layer normalization (pre-norm)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # Residual dropout
        self.dropout1 = nn.Dropout(resid_dropout)
        self.dropout2 = nn.Dropout(resid_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Multi-head attention with residual connection
        # Pre-norm: normalize before attention
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout1(attn_out)

        # Feed-forward with residual connection
        # Pre-norm: normalize before FFN
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)

        return x
