"""Dataset for token streams stored in sharded binary files."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import DTypeLike
from torch.utils.data import Dataset


class ShardedTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Dataset backed by a contiguous token stream split across shard files.

    The dataset treats all shard files as one logical token sequence and returns
    fixed-length (input, target) windows for next-token prediction.
    """

    def __init__(
        self,
        split_dir: str | Path,
        block_size: int,
        stride: int,
        dtype: DTypeLike,
        eos_token_id: int | None = None,
        mask_after_eos: bool = False,
        shard_entries: list[dict[str, Any]] | None = None,
        shard_cache_size: int = 2,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.block_size = block_size
        self.stride = stride
        self.dtype = np.dtype(dtype)
        self.eos_token_id = eos_token_id
        self.mask_after_eos = mask_after_eos
        self.shard_cache_size = max(1, shard_cache_size)
        self._memmaps: OrderedDict[int, np.memmap] = OrderedDict()

        if self.mask_after_eos and self.eos_token_id is None:
            raise ValueError("eos_token_id must be set when mask_after_eos=True")
        if self.block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if self.stride <= 0:
            raise ValueError("stride must be a positive integer")
        if self.stride > self.block_size:
            raise ValueError("stride cannot be larger than block_size")
        if not self.split_dir.exists():
            raise ValueError(f"Split directory does not exist: {self.split_dir}")
        if not self.split_dir.is_dir():
            raise ValueError(f"Split path is not a directory: {self.split_dir}")

        self._shards, self._shard_token_counts = self._resolve_shards(shard_entries)
        self._shard_starts = self._build_shard_starts(self._shard_token_counts)
        self._num_tokens = self._shard_starts[-1]

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

        start = idx * self.stride
        window = self._read_window(start=start, length=self.block_size + 1)
        x = torch.from_numpy(window[:-1].astype(np.int64))
        y = torch.from_numpy(window[1:].astype(np.int64))

        if self.mask_after_eos and self.eos_token_id is not None:
            eos_positions = (x == self.eos_token_id).nonzero(as_tuple=True)[0]
            if eos_positions.numel() > 0:
                first_eos = int(eos_positions[0])
                y[first_eos:] = -100

        return x, y

    def _resolve_shards(
        self, shard_entries: list[dict[str, Any]] | None
    ) -> tuple[list[Path], list[int]]:
        if shard_entries is None:
            shard_paths = sorted(self.split_dir.glob("part-*.bin"))
            if not shard_paths:
                raise ValueError(f"No shard files found in {self.split_dir}")
            token_counts = [self._tokens_for_file(path) for path in shard_paths]
            return shard_paths, token_counts

        shard_paths: list[Path] = []
        token_counts: list[int] = []
        for entry in shard_entries:
            if "file" not in entry:
                raise ValueError("Shard entry is missing 'file'")
            shard_path = self.split_dir / str(entry["file"])
            if not shard_path.exists():
                raise ValueError(f"Shard file does not exist: {shard_path}")
            if not shard_path.is_file():
                raise ValueError(f"Shard path is not a file: {shard_path}")
            actual_tokens = self._tokens_for_file(shard_path)
            manifest_tokens = int(entry.get("tokens", actual_tokens))
            if manifest_tokens != actual_tokens:
                raise ValueError(
                    f"Shard token count mismatch for {shard_path}: "
                    f"manifest={manifest_tokens}, actual={actual_tokens}"
                )
            shard_paths.append(shard_path)
            token_counts.append(actual_tokens)

        if not shard_paths:
            raise ValueError(f"No shard entries found for split in {self.split_dir}")

        return shard_paths, token_counts

    def _tokens_for_file(self, shard_path: Path) -> int:
        size = shard_path.stat().st_size
        if size % self.dtype.itemsize != 0:
            raise ValueError(
                f"File size {size} is not aligned with dtype size "
                f"{self.dtype.itemsize} for {shard_path}"
            )
        return size // self.dtype.itemsize

    def _build_shard_starts(self, token_counts: list[int]) -> list[int]:
        starts = [0]
        total = 0
        for count in token_counts:
            total += count
            starts.append(total)
        return starts

    def _find_shard_idx(self, token_pos: int) -> int:
        # Binary search over cumulative starts to locate containing shard.
        lo = 0
        hi = len(self._shard_starts) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._shard_starts[mid + 1] <= token_pos:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _get_memmap(self, shard_idx: int) -> np.memmap:
        memmap = self._memmaps.get(shard_idx)
        if memmap is not None:
            self._memmaps.move_to_end(shard_idx)
            return memmap

        memmap = np.memmap(self._shards[shard_idx], dtype=self.dtype, mode="r")
        self._memmaps[shard_idx] = memmap
        self._memmaps.move_to_end(shard_idx)
        if len(self._memmaps) > self.shard_cache_size:
            self._memmaps.popitem(last=False)
        return memmap

    def _read_window(self, start: int, length: int) -> np.ndarray[Any, Any]:
        remaining = length
        pos = start
        parts: list[np.ndarray[Any, Any]] = []

        while remaining > 0:
            shard_idx = self._find_shard_idx(pos)
            shard_start = self._shard_starts[shard_idx]
            shard_end = self._shard_starts[shard_idx + 1]
            in_shard_offset = pos - shard_start
            available = shard_end - pos
            take = min(remaining, available)

            shard_arr = self._get_memmap(shard_idx)
            part = shard_arr[in_shard_offset : in_shard_offset + take]
            parts.append(part)

            pos += take
            remaining -= take

        if len(parts) == 1:
            return np.asarray(parts[0], dtype=self.dtype)
        return np.concatenate(parts, dtype=self.dtype)
