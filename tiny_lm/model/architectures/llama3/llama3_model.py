"""Llama 3-style decoder-only language model.

This model uses:
- Token embeddings (no learned absolute positional embeddings)
- RoPE frequencies computed once at model level per forward pass
- A stack of Llama3Block modules
- Final RMSNorm + tied output projection
"""

import torch
import torch.nn as nn

from tiny_lm.model.normalization import RMSNorm
from tiny_lm.model.position import RoPE

from .llama3_block import Llama3Block


class Llama3(nn.Module):
    """Llama 3-style autoregressive language model.

    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        n_layers: Number of decoder blocks
        n_heads: Number of query heads
        context_length: Maximum sequence length
        n_kv_heads: Number of key/value heads (None -> n_heads)
        ffn_hidden_dim: SwiGLU hidden dimension (None -> 4 * d_model)
        multiple_of: Round FFN hidden dim up to this multiple
        rope_theta: Base frequency constant for RoPE
        norm_eps: Epsilon used in RMSNorm
        emb_dropout: Dropout on token embeddings
        attn_dropout: Dropout on attention weights
        resid_dropout: Dropout on residual branches
        ffn_dropout: Dropout inside SwiGLU gated representation
        qkv_bias: Whether to use bias in attention projections
        ffn_bias: Whether SwiGLU linear layers use bias
        attn_backend: Attention backend ("manual" or "sdp")
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        context_length: int = 1024,
        n_kv_heads: int | None = None,
        ffn_hidden_dim: int | None = None,
        multiple_of: int = 256,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-6,
        emb_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        qkv_bias: bool = False,
        ffn_bias: bool = False,
        attn_backend: str = "manual",
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if attn_backend not in ("manual", "sdp"):
            raise ValueError(
                f"attn_backend must be 'manual' or 'sdp', got {attn_backend}"
            )
        self.attn_backend = attn_backend

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.context_length = context_length

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.blocks = nn.ModuleList(
            [
                Llama3Block(
                    d_model=d_model,
                    n_heads=n_heads,
                    context_length=context_length,
                    n_kv_heads=n_kv_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                    attn_dropout=attn_dropout,
                    resid_dropout=resid_dropout,
                    ffn_dropout=ffn_dropout,
                    qkv_bias=qkv_bias,
                    ffn_bias=ffn_bias,
                    attn_backend=attn_backend,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm_f = RMSNorm(d_model, eps=norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # RoPE is generated once per forward and shared across blocks.
        self.rope = RoPE(
            dim=d_model // n_heads, max_seq_len=context_length, theta=rope_theta
        )

        # Weight tying between token embedding and output projection.
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear/embedding weights with GPT-style normal init."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token indices of shape (batch_size, seq_len)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        _, seq_len = input_ids.shape
        if seq_len > self.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum context length "
                f"{self.context_length}"
            )

        x = self.token_emb(input_ids)
        x = self.emb_dropout(x)

        freqs_cis = self.rope(seq_len)
        for block in self.blocks:
            x = block(x, freqs_cis=freqs_cis)

        x = self.norm_f(x)
        return self.lm_head(x)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_emb.weight.numel()
        return n_params
