"""Dataset for tokenized binary files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import DTypeLike
from torch.utils.data import Dataset


class BinTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Dataset backed by a contiguous token stream in a .bin file.

    Reads tokens via memmap for memory efficiency and creates fixed-length windows
    with configurable stride. Returns (input, target) pairs where target is input
    shifted by one token for next-token prediction.

    Training mode: samples windows that may cross document boundaries (EOS tokens).
    Validation mode: use mask_after_eos to ignore predictions beyond first EOS.

    Args:
        path: Path to the .bin file.
        block_size: Length of each input sequence (context window).
        stride: Step size between sequence start positions.
        dtype: Numpy dtype used in the .bin file.
        eos_token_id: Token id used as document boundary marker.
        mask_after_eos: If True, sets target tokens after first EOS to -100,
            preventing loss computation on cross-document predictions.
    """

    def __init__(
        self,
        path: str | Path,
        block_size: int,
        stride: int,
        dtype: DTypeLike,
        eos_token_id: int | None = None,
        mask_after_eos: bool = False,
    ) -> None:
        self.path = Path(path)
        self.block_size = block_size
        self.stride = stride
        self.dtype = np.dtype(dtype)
        self.eos_token_id = eos_token_id
        self.mask_after_eos = mask_after_eos
        self._tokens: np.memmap | None = None

        if self.mask_after_eos and self.eos_token_id is None:
            raise ValueError("eos_token_id must be set when mask_after_eos=True")
        if self.block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if self.stride <= 0:
            raise ValueError("stride must be a positive integer")
        if self.stride > self.block_size:
            raise ValueError("stride cannot be larger than block_size")

        file_size = self.path.stat().st_size
        if file_size % self.dtype.itemsize != 0:
            raise ValueError(
                f"File size {file_size} is not aligned with dtype size "
                f"{self.dtype.itemsize} for {self.path}"
            )
        self._num_tokens = file_size // self.dtype.itemsize

        if self._num_tokens < self.block_size + 1:
            raise ValueError(
                "Not enough tokens for the requested block_size. "
                f"Tokens={self._num_tokens}, block_size={self.block_size}"
            )

        max_start = self._num_tokens - (self.block_size + 1)
        self._num_windows = max_start // self.stride + 1

    def __len__(self) -> int:
        return self._num_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        if idx < 0 or idx >= self._num_windows:
            raise IndexError("Index out of range")

        if self._tokens is None:
            self._tokens = np.memmap(self.path, dtype=self.dtype, mode="r")

        start = idx * self.stride
        end = start + self.block_size
        x = torch.from_numpy(self._tokens[start:end].astype(np.int64))
        y = torch.from_numpy(self._tokens[start + 1 : end + 1].astype(np.int64))

        if self.mask_after_eos and self.eos_token_id is not None:
            eos_positions = (x == self.eos_token_id).nonzero(as_tuple=True)[0]
            if eos_positions.numel() > 0:
                first_eos = int(eos_positions[0])
                y[first_eos:] = -100

        return x, y
