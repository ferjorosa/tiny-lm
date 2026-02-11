"""Tests for Multi-Head Attention."""

import pytest
import torch
from tiny_lm.model.attention import MultiHeadAttention
from tiny_lm.model.position import RoPE


class TestMultiHeadAttention:
    """Test Multi-Head Attention mechanism."""

    def test_shape_preservation(self):
        """Test that attention preserves input shape."""
        d_model = 768
        n_heads = 12
        context_length = 1024

        attn = MultiHeadAttention(d_model, n_heads, context_length, dropout=0.0)

        # Test with various sequence lengths
        for seq_len in [1, 10, 50, 256]:
            x = torch.randn(2, seq_len, d_model)
            output = attn(x)

            assert output.shape == x.shape, (
                f"Expected shape {x.shape}, got {output.shape}"
            )

    def test_causal_masking(self):
        """Test that causal mask prevents attending to future tokens."""
        d_model = 64
        n_heads = 4
        context_length = 10

        attn = MultiHeadAttention(d_model, n_heads, context_length, dropout=0.0)

        # Create a sequence where each position has a unique value
        seq_len = 10
        x = (
            torch.arange(seq_len)
            .float()
            .view(1, seq_len, 1)
            .expand(1, seq_len, d_model)
        )

        # Get output
        output = attn(x)

        # First token should only see itself (position 0)
        # Last token should see all previous tokens (positions 0-9)
        # Due to causality, changing future tokens shouldn't affect past outputs

        # Modify the last token
        x_modified = x.clone()
        x_modified[:, -1, :] = 999.0

        output_modified = attn(x_modified)

        # All tokens except the last should have the same output
        # (they can't see the future modification)
        assert torch.allclose(
            output[:, :-1, :], output_modified[:, :-1, :], atol=1e-6
        ), "Causal mask should prevent attending to future tokens"

    def test_different_heads_different_outputs(self):
        """Test that using multiple heads is different from single head."""
        d_model = 64
        context_length = 128

        multi_head = MultiHeadAttention(
            d_model, n_heads=4, context_length=context_length, dropout=0.0
        )
        single_head = MultiHeadAttention(
            d_model, n_heads=1, context_length=context_length, dropout=0.0
        )

        x = torch.randn(2, 10, d_model)

        multi_output = multi_head(x)
        single_output = single_head(x)

        # Outputs should have same shape but likely different values
        assert multi_output.shape == single_output.shape
        # With random initialization, they should be different
        assert not torch.allclose(multi_output, single_output, atol=1e-3)

    def test_gradients_flow(self):
        """Test that gradients flow through all parameters."""
        d_model = 128
        n_heads = 8
        context_length = 512

        attn = MultiHeadAttention(d_model, n_heads, context_length, dropout=0.0)

        x = torch.randn(2, 10, d_model, requires_grad=True)
        output = attn(x)
        loss = output.sum()
        loss.backward()

        # Check input gradients
        assert x.grad is not None, "Gradients should flow through input"
        assert not torch.isnan(x.grad).any(), "Input gradients should be valid"

        # Check parameter gradients
        assert attn.W_q.weight.grad is not None, "Q projection should have gradients"
        assert attn.W_k.weight.grad is not None, "K projection should have gradients"
        assert attn.W_v.weight.grad is not None, "V projection should have gradients"
        assert attn.out_proj.weight.grad is not None, (
            "Output projection should have gradients"
        )

    def test_attention_with_bias(self):
        """Test attention with QKV bias enabled."""
        d_model = 256
        n_heads = 8
        context_length = 512

        attn_with_bias = MultiHeadAttention(
            d_model, n_heads, context_length, dropout=0.0, qkv_bias=True
        )
        attn_no_bias = MultiHeadAttention(
            d_model, n_heads, context_length, dropout=0.0, qkv_bias=False
        )

        # Check that bias exists when enabled
        assert attn_with_bias.W_q.bias is not None
        assert attn_with_bias.W_k.bias is not None
        assert attn_with_bias.W_v.bias is not None

        # Check that bias doesn't exist when disabled
        assert attn_no_bias.W_q.bias is None
        assert attn_no_bias.W_k.bias is None
        assert attn_no_bias.W_v.bias is None

    def test_batch_independence(self):
        """Test that different batch elements are processed independently."""
        d_model = 128
        n_heads = 4
        context_length = 256

        attn = MultiHeadAttention(d_model, n_heads, context_length, dropout=0.0)

        # Create batch with different sequences
        x1 = torch.randn(1, 10, d_model)
        x2 = torch.randn(1, 10, d_model)
        x_batch = torch.cat([x1, x2], dim=0)

        # Process individually and in batch
        out1 = attn(x1)
        out2 = attn(x2)
        out_batch = attn(x_batch)

        # Results should match
        assert torch.allclose(out1, out_batch[0:1], atol=1e-6), (
            "First batch element should match individual processing"
        )
        assert torch.allclose(out2, out_batch[1:2], atol=1e-6), (
            "Second batch element should match individual processing"
        )

    def test_supports_grouped_query_attention(self):
        """Test GQA mode with fewer KV heads than query heads."""
        d_model = 128
        n_heads = 8
        n_kv_heads = 2
        context_length = 128

        attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            context_length=context_length,
            dropout=0.0,
        )
        x = torch.randn(2, 16, d_model)
        out = attn(x)

        assert out.shape == x.shape, "GQA should preserve input shape"
        assert attn.W_q.weight.shape == (n_heads * attn.head_dim, d_model)
        assert attn.W_k.weight.shape == (n_kv_heads * attn.head_dim, d_model)
        assert attn.W_v.weight.shape == (n_kv_heads * attn.head_dim, d_model)
        assert attn.n_rep == n_heads // n_kv_heads

    def test_supports_multi_query_attention(self):
        """Test MQA mode with a single KV head."""
        d_model = 128
        n_heads = 8
        context_length = 128

        attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=1,
            context_length=context_length,
            dropout=0.0,
        )
        x = torch.randn(2, 16, d_model)
        out = attn(x)

        assert out.shape == x.shape, "MQA should preserve input shape"
        assert attn.W_k.weight.shape == (attn.head_dim, d_model)
        assert attn.W_v.weight.shape == (attn.head_dim, d_model)

    def test_invalid_kv_head_configuration_raises(self):
        """Test that invalid n_kv_heads config is rejected."""
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadAttention(
                d_model=128,
                n_heads=8,
                n_kv_heads=3,
                context_length=128,
                dropout=0.0,
            )

    def test_rope_is_optional_and_preserves_shape(self):
        """Test optional RoPE path runs and preserves output shape."""
        d_model = 128
        n_heads = 8
        context_length = 128

        attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            context_length=context_length,
            dropout=0.0,
        )
        rope = RoPE(dim=d_model // n_heads, max_seq_len=context_length)
        x = torch.randn(2, 16, d_model)
        out = attn(x, freqs_cis=rope(seq_len=16))

        assert out.shape == x.shape, "RoPE-enabled attention should preserve shape"

    def test_rope_matches_non_rope_for_single_token(self):
        """Test RoPE and non-RoPE are equivalent at position 0 only."""
        d_model = 128
        n_heads = 8
        context_length = 32

        attn_no_rope = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            context_length=context_length,
            dropout=0.0,
        )
        attn_rope = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            context_length=context_length,
            dropout=0.0,
        )
        rope = RoPE(dim=d_model // n_heads, max_seq_len=context_length)

        # Keep the same projection weights so differences only come from RoPE.
        with torch.no_grad():
            attn_rope.W_q.weight.copy_(attn_no_rope.W_q.weight)
            attn_rope.W_k.weight.copy_(attn_no_rope.W_k.weight)
            attn_rope.W_v.weight.copy_(attn_no_rope.W_v.weight)
            attn_rope.out_proj.weight.copy_(attn_no_rope.out_proj.weight)
            attn_rope.out_proj.bias.copy_(attn_no_rope.out_proj.bias)

        x = torch.randn(2, 1, d_model)
        out_no_rope = attn_no_rope(x)
        out_rope = attn_rope(x, freqs_cis=rope(seq_len=1))

        assert torch.allclose(out_no_rope, out_rope, atol=1e-6), (
            "For seq_len=1, RoPE should be identity and match non-RoPE output"
        )
