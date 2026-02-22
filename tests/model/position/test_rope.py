"""Tests for rotary positional embeddings (RoPE)."""

import pytest
import torch

from tiny_lm.model.components.position import RoPE, apply_rotary_emb


class TestRoPE:
    """Test RoPE frequency generation and rotary application."""

    def test_output_shape_and_type(self):
        """Test generated frequency table shape and complex type."""
        rope = RoPE(dim=64, max_seq_len=512)
        freqs = rope(seq_len=32)

        assert freqs.shape == (32, 32), "Expected shape (seq_len, dim // 2)"
        assert torch.is_complex(freqs), "RoPE frequencies should be complex"

    def test_deterministic_outputs(self):
        """Test that same sequence length returns identical frequencies."""
        rope = RoPE(dim=32, max_seq_len=128)
        f1 = rope(seq_len=20)
        f2 = rope(seq_len=20)

        assert torch.allclose(f1, f2), "RoPE frequencies should be deterministic"

    def test_exceeds_max_length_raises_error(self):
        """Test that requesting seq_len beyond max_seq_len raises ValueError."""
        rope = RoPE(dim=32, max_seq_len=16)

        with pytest.raises(ValueError, match="exceeds maximum"):
            rope(seq_len=17)

    def test_apply_rotary_emb_preserves_shape_and_dtype(self):
        """Test that applying RoPE keeps tensor shape and dtype unchanged."""
        batch_size, seq_len = 2, 10
        n_heads, n_kv_heads, head_dim = 8, 4, 64

        rope = RoPE(dim=head_dim, max_seq_len=128)
        freqs = rope(seq_len)

        xq = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=torch.float32)
        xk = torch.randn(batch_size, seq_len, n_kv_heads, head_dim, dtype=torch.float32)

        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs)

        assert xq_out.shape == xq.shape, "xq shape should be preserved"
        assert xk_out.shape == xk.shape, "xk shape should be preserved"
        assert xq_out.dtype == xq.dtype, "xq dtype should be preserved"
        assert xk_out.dtype == xk.dtype, "xk dtype should be preserved"

    def test_apply_rotary_emb_identity_for_position_zero(self):
        """Test that seq_len=1 (position 0 only) leaves vectors unchanged."""
        batch_size, seq_len = 2, 1
        n_heads, n_kv_heads, head_dim = 6, 3, 32

        rope = RoPE(dim=head_dim, max_seq_len=8)
        freqs = rope(seq_len)

        xq = torch.randn(batch_size, seq_len, n_heads, head_dim)
        xk = torch.randn(batch_size, seq_len, n_kv_heads, head_dim)

        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs)

        assert torch.allclose(xq_out, xq, atol=1e-6), (
            "Position-0 rotation should be identity for xq"
        )
        assert torch.allclose(xk_out, xk, atol=1e-6), (
            "Position-0 rotation should be identity for xk"
        )

    def test_apply_rotary_emb_preserves_l2_norm(self):
        """Test that per-vector norms are preserved by rotation."""
        batch_size, seq_len = 2, 12
        n_heads, n_kv_heads, head_dim = 4, 2, 64

        rope = RoPE(dim=head_dim, max_seq_len=128)
        freqs = rope(seq_len)

        xq = torch.randn(batch_size, seq_len, n_heads, head_dim)
        xk = torch.randn(batch_size, seq_len, n_kv_heads, head_dim)

        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs)

        xq_norm = xq.norm(dim=-1)
        xk_norm = xk.norm(dim=-1)
        xq_out_norm = xq_out.norm(dim=-1)
        xk_out_norm = xk_out.norm(dim=-1)

        assert torch.allclose(xq_norm, xq_out_norm, atol=1e-5), (
            "RoPE should preserve xq L2 norms"
        )
        assert torch.allclose(xk_norm, xk_out_norm, atol=1e-5), (
            "RoPE should preserve xk L2 norms"
        )
