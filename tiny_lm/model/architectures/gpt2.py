"""GPT-2 Model Architecture.

A decoder-only transformer for causal language modeling. The model predicts the next
token given all previous tokens in the sequence.

Paper: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
"""

import torch
import torch.nn as nn
from tiny_lm.model.activation import GELU
from tiny_lm.model.blocks import GPT2Block
from tiny_lm.model.normalization import LayerNorm
from tiny_lm.model.position import LearnedPositionalEmbedding


class GPT2(nn.Module):
    """GPT-2 Language Model.

    A decoder-only transformer for autoregressive language modeling.
    Uses pre-norm architecture and causal masking.

    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        n_layers: Number of transformer blocks
        n_heads: Number of attention heads per block
        d_ff: Feed-forward hidden dimension
        context_length: Maximum sequence length
        emb_dropout: Dropout for embeddings
        attn_dropout: Dropout for attention weights
        resid_dropout: Dropout for residual connections
        ffn_dropout: Dropout in feed-forward networks
        qkv_bias: Whether to use bias in attention QKV projections
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        context_length: int = 1024,
        emb_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.context_length = context_length

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Positional embeddings
        self.pos_emb = LearnedPositionalEmbedding(context_length, d_model)

        # Embedding dropout
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                GPT2Block(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    context_length=context_length,
                    activation=GELU(),
                    attn_dropout=attn_dropout,
                    resid_dropout=resid_dropout,
                    ffn_dropout=ffn_dropout,
                    qkv_bias=qkv_bias,
                )
                for _ in range(n_layers)
            ]
        )

        # Final layer normalization
        self.ln_f = LayerNorm(d_model)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share weights between token embeddings and output projection
        # This reduces parameters and often improves performance
        self.lm_head.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following GPT-2 paper.

        - Linear layers: normal distribution with std=0.02
        - Embeddings: normal distribution with std=0.02
        - Biases: zeros
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through GPT-2 model.

        Args:
            input_ids: Token indices of shape (batch_size, seq_len)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Check sequence length
        if seq_len > self.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum "
                f"context length {self.context_length}"
            )

        # Get token embeddings: (batch_size, seq_len, d_model)
        token_embeddings = self.token_emb(input_ids)

        # Get positional embeddings: (seq_len, d_model)
        pos_embeddings = self.pos_emb(input_ids)

        # Combine embeddings: (batch_size, seq_len, d_model)
        x = token_embeddings + pos_embeddings
        x = self.emb_dropout(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer normalization
        x = self.ln_f(x)

        # Project to vocabulary: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)

        return logits

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters in the model.

        Args:
            non_embedding: If True, exclude embedding parameters

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_emb.weight.numel()
            n_params -= self.pos_emb.pos_emb.weight.numel()
        return n_params
