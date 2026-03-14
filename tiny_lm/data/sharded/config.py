"""Configuration for sharded token stream data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ShardedDataConfig:
    """Configuration for sharded token datasets and data modules."""

    data_root: str | None
    manifest_path: str | None
    train_split: str
    val_split: str
    train_dir: str | None
    val_dir: str | None
    block_size: int
    stride: int
    dtype: str
    eos_token_id: int | None
    batch_size: int
    num_workers: int
    pin_memory: bool
    drop_last: bool
    format: str = "sharded_bin_v1"

    def __post_init__(self) -> None:
        if (self.data_root is None) == (self.manifest_path is None):
            raise ValueError("Exactly one of data_root or manifest_path must be set")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.stride > self.block_size:
            raise ValueError("stride cannot be larger than block_size")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.format != "sharded_bin_v1":
            raise ValueError(f"Unsupported format: {self.format}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ShardedDataConfig":
        """Load config from YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def resolve_data_root(self) -> Path:
        """Resolve and validate the tokenized data root directory."""
        if self.data_root is not None:
            root = Path(self.data_root)
        elif self.manifest_path is not None:
            root = Path(self.manifest_path).parent
        else:
            raise RuntimeError("Invalid config state: no data root source set")
        if not root.exists():
            raise ValueError(f"Data root does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Data root is not a directory: {root}")
        return root

    def resolve_manifest_path(self) -> Path:
        """Resolve and validate the sharded manifest path."""
        if self.manifest_path is not None:
            manifest = Path(self.manifest_path)
        else:
            manifest = self.resolve_data_root() / "manifest.json"
        if not manifest.exists():
            raise ValueError(f"Manifest file does not exist: {manifest}")
        if not manifest.is_file():
            raise ValueError(f"Manifest path is not a file: {manifest}")
        return manifest

    def split_dir(self, split: str) -> Path:
        """Return the directory that stores shard files for a split."""
        root = self.resolve_data_root()
        if split == "train" and self.train_dir is not None:
            split_dir = Path(self.train_dir)
        elif split == "val" and self.val_dir is not None:
            split_dir = Path(self.val_dir)
        elif split == "train":
            split_dir = root / self.train_split
        elif split == "val":
            split_dir = root / self.val_split
        else:
            raise ValueError(f"Unknown split: {split}")
        if not split_dir.exists():
            raise ValueError(f"Split directory does not exist for {split}: {split_dir}")
        if not split_dir.is_dir():
            raise ValueError(f"Split path is not a directory for {split}: {split_dir}")
        return split_dir

    def split_name(self, split: str) -> str:
        """Return split key used in manifest.json for a split."""
        if split == "train":
            return self.train_split
        if split == "val":
            return self.val_split
        raise ValueError(f"Unknown split: {split}")

    def load_manifest(self) -> dict[str, Any]:
        """Load and return manifest JSON contents."""
        import json

        manifest_path = self.resolve_manifest_path()
        with open(manifest_path) as f:
            manifest = json.load(f)
        return manifest
