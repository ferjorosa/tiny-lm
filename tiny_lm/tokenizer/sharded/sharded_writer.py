"""Buffered shard writer utilities for token streams."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class ShardedBinWriter:
    """Buffered writer that splits a token stream across part-*.bin shards."""

    def __init__(
        self,
        split_dir: Path,
        dtype: np.dtype[Any],
        tokens_per_shard: int,
        flush_tokens: int,
    ) -> None:
        self.split_dir = split_dir
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype
        self.tokens_per_shard = tokens_per_shard
        self.flush_tokens = flush_tokens
        self.buffer: list[np.ndarray[Any, Any]] = []
        self.buffer_tokens = 0

        self.current_shard_idx = 0
        self.current_shard_tokens = 0
        self.current_handle = self._open_shard(self.current_shard_idx)
        self.shards: list[dict[str, int | str]] = []

    def _open_shard(self, idx: int):
        shard_path = self.split_dir / f"part-{idx:05d}.bin"
        return open(shard_path, "wb")

    def _finalize_current_shard(self) -> None:
        if self.current_shard_tokens == 0:
            return
        shard_name = f"part-{self.current_shard_idx:05d}.bin"
        self.shards.append({"file": shard_name, "tokens": self.current_shard_tokens})
        self.current_handle.close()
        self.current_shard_idx += 1
        self.current_shard_tokens = 0
        self.current_handle = self._open_shard(self.current_shard_idx)

    def _write_array(self, arr: np.ndarray[Any, Any]) -> None:
        offset = 0
        while offset < arr.size:
            remaining = self.tokens_per_shard - self.current_shard_tokens
            take = min(remaining, arr.size - offset)
            chunk = arr[offset : offset + take]
            chunk.tofile(self.current_handle)
            self.current_shard_tokens += take
            offset += take
            if self.current_shard_tokens == self.tokens_per_shard:
                self._finalize_current_shard()

    def add(self, arr: np.ndarray[Any, Any]) -> None:
        if arr.size == 0:
            return
        self.buffer.append(arr)
        self.buffer_tokens += arr.size
        if self.buffer_tokens >= self.flush_tokens:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        if len(self.buffer) == 1:
            merged = self.buffer[0]
        else:
            merged = np.concatenate(self.buffer, dtype=self.dtype)
        self._write_array(merged)
        self.buffer = []
        self.buffer_tokens = 0

    def close(self) -> list[dict[str, int | str]]:
        self.flush()
        self.current_handle.close()
        if self.current_shard_tokens > 0:
            shard_name = f"part-{self.current_shard_idx:05d}.bin"
            self.shards.append({"file": shard_name, "tokens": self.current_shard_tokens})
        else:
            # Remove trailing empty shard file that can be created after rolling over.
            empty_shard = self.split_dir / f"part-{self.current_shard_idx:05d}.bin"
            if empty_shard.exists():
                empty_shard.unlink()
        return self.shards


def prepare_split_dir(split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for part_file in split_dir.glob("part-*.bin"):
        part_file.unlink()
