"""Tests for standard feed-forward network."""

import torch
from tiny_lm.model.components.feedforward import FeedForward
from tiny_lm.model.components.activation import GELU


class TestFeedForward:
    """Test standard feed-forward network."""

    def test_shape_preservation(self):
        """Test that FFN preserves batch and sequence dimensions."""
        d_model = 768
        d_ff = 3072  # 4 * d_model

        ffn = FeedForward(d_model, d_ff, activation=GELU(), dropout=0.0)

        # Test with various shapes
        test_cases = [
            (2, 1, d_model),  # Single token
            (2, 10, d_model),  # Short sequence
            (4, 100, d_model),  # Longer sequence
        ]

        for shape in test_cases:
            x = torch.randn(*shape)
            output = ffn(x)

            assert output.shape == x.shape, (
                f"Expected shape {x.shape}, got {output.shape}"
            )

    def test_expansion_factor(self):
        """Test that hidden dimension is correctly used."""
        d_model = 256
        d_ff = 1024  # 4x expansion

        ffn = FeedForward(d_model, d_ff, activation=GELU(), dropout=0.0)

        # Check layer dimensions
        assert ffn.fc1.weight.shape == (d_ff, d_model), (
            f"First layer should expand to {d_ff}"
        )
        assert ffn.fc2.weight.shape == (d_model, d_ff), (
            f"Second layer should contract to {d_model}"
        )

    def test_gradients_flow(self):
        """Test that gradients flow through all parameters."""
        d_model = 128
        d_ff = 512

        ffn = FeedForward(d_model, d_ff, activation=GELU(), dropout=0.0)

        x = torch.randn(2, 10, d_model, requires_grad=True)
        output = ffn(x)
        loss = output.sum()
        loss.backward()

        # Check input gradients
        assert x.grad is not None, "Gradients should flow through input"
        assert not torch.isnan(x.grad).any(), "Input gradients should be valid"

        # Check parameter gradients
        assert ffn.fc1.weight.grad is not None, "First layer should have gradients"
        assert ffn.fc1.bias.grad is not None, "First layer bias should have gradients"
        assert ffn.fc2.weight.grad is not None, "Second layer should have gradients"
        assert ffn.fc2.bias.grad is not None, "Second layer bias should have gradients"

    def test_dropout_behavior(self):
        """Test that dropout works during training but not during eval."""
        d_model = 256
        d_ff = 1024

        ffn = FeedForward(d_model, d_ff, activation=GELU(), dropout=0.5)
        x = torch.randn(2, 10, d_model)

        # Training mode: outputs should differ (due to dropout randomness)
        ffn.train()
        out1 = ffn(x)
        out2 = ffn(x)
        assert not torch.allclose(out1, out2, atol=1e-6), (
            "Dropout should cause different outputs in training mode"
        )

        # Eval mode: outputs should be identical
        ffn.eval()
        out1 = ffn(x)
        out2 = ffn(x)
        assert torch.allclose(out1, out2), (
            "Outputs should be deterministic in eval mode"
        )

    def test_different_expansion_factors(self):
        """Test FFN with different expansion factors."""
        d_model = 128

        for expansion in [2, 4, 8]:
            d_ff = d_model * expansion
            ffn = FeedForward(d_model, d_ff, activation=GELU(), dropout=0.0)

            x = torch.randn(2, 10, d_model)
            output = ffn(x)

            assert output.shape == x.shape, (
                f"Shape should be preserved with {expansion}x expansion"
            )

    def test_batch_independence(self):
        """Test that different batch elements are processed independently."""
        d_model = 256
        d_ff = 1024

        ffn = FeedForward(d_model, d_ff, activation=GELU(), dropout=0.0)

        # Create batch with different sequences
        x1 = torch.randn(1, 10, d_model)
        x2 = torch.randn(1, 10, d_model)
        x_batch = torch.cat([x1, x2], dim=0)

        # Process individually and in batch
        out1 = ffn(x1)
        out2 = ffn(x2)
        out_batch = ffn(x_batch)

        # Results should match
        assert torch.allclose(out1, out_batch[0:1], atol=1e-6), (
            "First batch element should match individual processing"
        )
        assert torch.allclose(out2, out_batch[1:2], atol=1e-6), (
            "Second batch element should match individual processing"
        )

    def test_custom_activation(self):
        """Test FFN with different activation functions."""
        d_model = 128
        d_ff = 512
        x = torch.randn(2, 10, d_model)

        # Test with ReLU
        ffn_relu = FeedForward(d_model, d_ff, dropout=0.0, activation=torch.nn.ReLU())
        out_relu = ffn_relu(x)
        assert out_relu.shape == x.shape, "ReLU FFN should preserve shape"

        # Test with GELU
        ffn_gelu = FeedForward(d_model, d_ff, activation=GELU(), dropout=0.0)
        out_gelu = ffn_gelu(x)
        assert out_gelu.shape == x.shape, "GELU FFN should preserve shape"

        # Different activations should produce different outputs
        assert not torch.allclose(out_relu, out_gelu, atol=1e-3), (
            "Different activations should produce different results"
        )
