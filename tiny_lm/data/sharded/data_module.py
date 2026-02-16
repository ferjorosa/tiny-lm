"""Lightning data module for tokenized sharded binary files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
from numpy.typing import DTypeLike
from torch.utils.data import DataLoader

from tiny_lm.data.sharded.dataset import ShardedTokenDataset


class ShardedTokenDataModule(pl.LightningDataModule):
    """
    DataModule for training on token streams stored as shard directories.

    Args:
        train_dir: Path to train shard directory.
        val_dir: Path to validation shard directory.
        block_size: Sequence length for each sample.
        stride: Step size between sequence start positions.
        dtype: Numpy dtype used in shard files.
        eos_token_id: Token id used as document boundary marker.
        batch_size: Number of sequences per batch.
        num_workers: Number of DataLoader workers.
        pin_memory: Whether to pin memory in DataLoader.
        drop_last: Whether to drop last partial batch.
        train_shards: Optional manifest entries for train split.
        val_shards: Optional manifest entries for val split.
    """

    def __init__(
        self,
        train_dir: str | Path,
        val_dir: str | Path,
        block_size: int,
        stride: int,
        dtype: DTypeLike,
        eos_token_id: int | None,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        train_shards: list[dict[str, Any]] | None = None,
        val_shards: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.block_size = block_size
        self.stride = stride
        self.dtype = np.dtype(dtype)
        self.eos_token_id = eos_token_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.train_shards = train_shards
        self.val_shards = val_shards

        self.train_dataset: ShardedTokenDataset | None = None
        self.val_dataset: ShardedTokenDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = ShardedTokenDataset(
                split_dir=self.train_dir,
                block_size=self.block_size,
                stride=self.stride,
                dtype=self.dtype,
                shard_entries=self.train_shards,
            )
            self.val_dataset = ShardedTokenDataset(
                split_dir=self.val_dir,
                block_size=self.block_size,
                stride=self.stride,
                dtype=self.dtype,
                eos_token_id=self.eos_token_id,
                mask_after_eos=True,
                shard_entries=self.val_shards,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("setup() must be called before train_dataloader()")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("setup() must be called before val_dataloader()")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=False,
        )
