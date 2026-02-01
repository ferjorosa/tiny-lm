"""Tests for learned positional embeddings."""

import torch
import pytest
from tiny_lm.model.position import LearnedPositionalEmbedding


class TestLearnedPositionalEmbedding:
    """Test learned positional embeddings."""

    def test_output_shape(self):
        """Test that output shape is (seq_len, d_model)."""
        max_seq_len = 1024
        d_model = 768
        pos_emb = LearnedPositionalEmbedding(max_seq_len, d_model)

        # Test with various sequence lengths
        for seq_len in [1, 10, 100, 512]:
            x = torch.randint(0, 1000, (2, seq_len))
            output = pos_emb(x)

            assert output.shape == (seq_len, d_model), (
                f"Expected shape ({seq_len}, {d_model}), got {output.shape}"
            )

    def test_deterministic(self):
        """Test that same positions always produce same embeddings."""
        max_seq_len = 512
        d_model = 256
        pos_emb = LearnedPositionalEmbedding(max_seq_len, d_model)

        x = torch.randint(0, 100, (4, 10))

        # Get embeddings multiple times
        emb1 = pos_emb(x)
        emb2 = pos_emb(x)
        emb3 = pos_emb(x)

        assert torch.allclose(emb1, emb2), "Embeddings should be deterministic"
        assert torch.allclose(emb2, emb3), "Embeddings should be deterministic"

    def test_different_positions_different_embeddings(self):
        """Test that different positions have different embeddings."""
        max_seq_len = 512
        d_model = 256
        pos_emb = LearnedPositionalEmbedding(max_seq_len, d_model)

        # Get embeddings for a sequence
        x = torch.randint(0, 100, (1, 10))
        embeddings = pos_emb(x)

        # Check that consecutive positions are different
        for i in range(embeddings.shape[0] - 1):
            assert not torch.allclose(embeddings[i], embeddings[i + 1]), (
                f"Position {i} and {i + 1} should have different embeddings"
            )

    def test_matches_pytorch_embedding(self):
        """Test that implementation matches PyTorch's Embedding."""
        max_seq_len = 256
        d_model = 128

        custom_pos = LearnedPositionalEmbedding(max_seq_len, d_model)
        pytorch_emb = torch.nn.Embedding(max_seq_len, d_model)

        # Copy weights to ensure same initialization
        with torch.no_grad():
            pytorch_emb.weight.copy_(custom_pos.pos_emb.weight)

        x = torch.randint(0, 100, (2, 10))
        positions = torch.arange(10)

        custom_out = custom_pos(x)
        pytorch_out = pytorch_emb(positions)

        assert torch.allclose(custom_out, pytorch_out), (
            "Custom implementation should match PyTorch Embedding"
        )

    def test_exceeds_max_length_raises_error(self):
        """Test that exceeding max_seq_len raises an error."""
        max_seq_len = 64
        d_model = 128
        pos_emb = LearnedPositionalEmbedding(max_seq_len, d_model)

        # Try with sequence longer than max
        x = torch.randint(0, 100, (2, max_seq_len + 10))

        with pytest.raises(ValueError, match="exceeds maximum"):
            pos_emb(x)

    def test_gradients_flow(self):
        """Test that gradients flow through positional embeddings."""
        max_seq_len = 256
        d_model = 128
        pos_emb = LearnedPositionalEmbedding(max_seq_len, d_model)

        x = torch.randint(0, 100, (2, 10))
        y = pos_emb(x)
        loss = y.sum()
        loss.backward()

        assert pos_emb.pos_emb.weight.grad is not None, (
            "Gradients should flow through embeddings"
        )
        assert not torch.isnan(pos_emb.pos_emb.weight.grad).any(), (
            "Gradients should be valid"
        )
