"""Dataclasses used by the sharded tokenization pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SplitWriteResult:
    token_count: int
    example_count: int
    dtype: np.dtype[Any]
    length_stats: dict[str, dict[str, float]]
    shards: list[dict[str, int | str]]


@dataclass
class BatchTokenizeResult:
    arrays: list[np.ndarray[Any, Any]]
    token_count: int
    example_count: int
    length_stats_raw: dict[str, int]
