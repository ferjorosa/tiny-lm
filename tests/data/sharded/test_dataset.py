"""Tests for ShardedTokenDataset."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tiny_lm.data.sharded import ShardedTokenDataset


def _write_shard(split_dir, index: int, tokens: np.ndarray) -> None:
    shard_path = split_dir / f"part-{index:05d}.bin"
    tokens.tofile(shard_path)


class TestShardedTokenDataset:
    """Tests for sharded token dataset."""

    def test_length_and_cross_shard_shift(self, tmp_path):
        train_dir = tmp_path / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        _write_shard(train_dir, 0, np.array([1, 2, 3], dtype=np.uint16))
        _write_shard(train_dir, 1, np.array([4, 5, 6, 7], dtype=np.uint16))

        dataset = ShardedTokenDataset(
            split_dir=train_dir,
            block_size=4,
            stride=2,
            dtype=np.uint16,
        )

        assert len(dataset) == 2

        x0, y0 = dataset[0]
        assert torch.equal(x0, torch.tensor([1, 2, 3, 4]))
        assert torch.equal(y0, torch.tensor([2, 3, 4, 5]))

        x1, y1 = dataset[1]
        assert torch.equal(x1, torch.tensor([3, 4, 5, 6]))
        assert torch.equal(y1, torch.tensor([4, 5, 6, 7]))

    def test_mask_after_eos(self, tmp_path):
        val_dir = tmp_path / "val"
        val_dir.mkdir(parents=True, exist_ok=True)
        _write_shard(val_dir, 0, np.array([10, 11], dtype=np.uint16))
        _write_shard(val_dir, 1, np.array([3, 20, 21, 22], dtype=np.uint16))

        dataset = ShardedTokenDataset(
            split_dir=val_dir,
            block_size=4,
            stride=1,
            dtype=np.uint16,
            eos_token_id=3,
            mask_after_eos=True,
        )

        x, y = dataset[0]
        assert torch.equal(x, torch.tensor([10, 11, 3, 20]))
        assert torch.equal(y, torch.tensor([11, 3, -100, -100]))

    def test_manifest_shard_tokens_must_match_file(self, tmp_path):
        train_dir = tmp_path / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        _write_shard(train_dir, 0, np.array([1, 2, 3, 4, 5], dtype=np.uint16))

        with pytest.raises(ValueError, match="Shard token count mismatch"):
            ShardedTokenDataset(
                split_dir=train_dir,
                block_size=2,
                stride=1,
                dtype=np.uint16,
                shard_entries=[{"file": "part-00000.bin", "tokens": 999}],
            )

    def test_raises_when_not_enough_tokens(self, tmp_path):
        train_dir = tmp_path / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        _write_shard(train_dir, 0, np.array([1, 2], dtype=np.uint16))

        with pytest.raises(ValueError, match="Not enough tokens"):
            ShardedTokenDataset(
                split_dir=train_dir,
                block_size=2,
                stride=1,
                dtype=np.uint16,
            )
