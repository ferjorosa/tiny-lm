"""Lightning data module for tokenized binary files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from numpy.typing import DTypeLike
from torch.utils.data import DataLoader

from tiny_lm.data.bin.dataset import BinTokenDataset


class BinTokenDataModule(pl.LightningDataModule):
    """
    DataModule for training on token streams stored as .bin files.

    Note: No custom collate_fn is required because each dataset item already
    returns fixed-shape tensors (x, y). The default collate stacks them into
    [batch_size, block_size].

    Args:
        train_path: Path to train .bin file.
        val_path: Path to validation .bin file.
        block_size: Sequence length for each sample.
        stride: Step size between sequence start positions.
        dtype: Numpy dtype used in the .bin file.
        eos_token_id: Token id used as document boundary marker.
        batch_size: Number of sequences per batch.
        num_workers: Number of DataLoader workers.
        pin_memory: Whether to pin memory in DataLoader.
        drop_last: Whether to drop last partial batch.
    """

    def __init__(
        self,
        train_path: str | Path,
        val_path: str | Path,
        block_size: int,
        stride: int,
        dtype: DTypeLike,
        eos_token_id: int | None,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.block_size = block_size
        self.stride = stride
        self.dtype = np.dtype(dtype)
        self.eos_token_id = eos_token_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.train_dataset: BinTokenDataset | None = None
        self.val_dataset: BinTokenDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = BinTokenDataset(
                path=self.train_path,
                block_size=self.block_size,
                stride=self.stride,
                dtype=self.dtype,
            )
            self.val_dataset = BinTokenDataset(
                path=self.val_path,
                block_size=self.block_size,
                stride=self.stride,
                dtype=self.dtype,
                eos_token_id=self.eos_token_id,
                mask_after_eos=True,
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
