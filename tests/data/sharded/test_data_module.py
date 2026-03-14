"""Smoke tests for ShardedTokenDataModule."""

from __future__ import annotations

import numpy as np

from tiny_lm.data.sharded import ShardedTokenDataModule


def _write_shard(split_dir, index: int, tokens: np.ndarray) -> None:
    shard_path = split_dir / f"part-{index:05d}.bin"
    tokens.tofile(shard_path)


def test_sharded_data_module_train_batch_shape(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    _write_shard(train_dir, 0, np.arange(1, 33, dtype=np.uint16))
    _write_shard(val_dir, 0, np.arange(1, 33, dtype=np.uint16))

    data_module = ShardedTokenDataModule(
        train_dir=train_dir,
        val_dir=val_dir,
        block_size=8,
        stride=8,
        dtype=np.uint16,
        eos_token_id=3,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    data_module.setup("fit")

    x, y = next(iter(data_module.train_dataloader()))
    assert tuple(x.shape) == (2, 8)
    assert tuple(y.shape) == (2, 8)
