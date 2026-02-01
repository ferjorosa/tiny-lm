"""Learned positional embeddings.

Learned positional embeddings use a trainable lookup table where each position
in the sequence has its own learnable embedding vector that's added to the token
embeddings.

This approach is simpler than sinusoidal encodings but requires more parameters.
It was first introduced in GPT-1 and continued in GPT-2.

Paper: "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
"""

import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings for GPT-2.

    Creates a lookup table of position embeddings that are learned during training.
    Each position from 0 to max_seq_len-1 has its own embedding vector.

    Args:
        max_seq_len: Maximum sequence length the model can handle
        d_model: Dimension of the embeddings
    """

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Learnable position embeddings
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get positional embeddings for input sequence.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
               or (batch_size, seq_len) for token indices

        Returns:
            Positional embeddings of shape (seq_len, d_model)
            These should be added to token embeddings: tokens_emb + pos_emb
        """
        batch_size, seq_len = x.shape[:2]

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum "
                f"sequence length {self.max_seq_len}"
            )

        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device)

        # Get position embeddings (seq_len, d_model)
        return self.pos_emb(positions)
