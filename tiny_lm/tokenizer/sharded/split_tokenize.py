"""Parallel split tokenization and ordered shard writing helpers."""

from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from .models import BatchTokenizeResult, SplitWriteResult
from .sharded_writer import ShardedBinWriter, prepare_split_dir
from .stats import (
    finalize_length_stats,
    init_length_stats,
    merge_length_stats,
    update_length_stats,
)


def _tokenize_text_batch(
    texts: list[str],
    tokenizer: Any,
    bos_token_id: int,
    eos_token_id: int,
    dtype: np.dtype[Any],
) -> BatchTokenizeResult:
    arrays: list[np.ndarray[Any, Any]] = []
    token_count = 0
    example_count = 0
    length_stats = init_length_stats()

    for text in texts:
        if text == "":
            continue
        ids = [bos_token_id] + tokenizer.encode_ordinary(text) + [eos_token_id]
        arr = np.asarray(ids, dtype=dtype)
        arrays.append(arr)
        token_count += arr.size
        example_count += 1
        update_length_stats(length_stats, len(text), arr.size)

    return BatchTokenizeResult(
        arrays=arrays,
        token_count=token_count,
        example_count=example_count,
        length_stats_raw=length_stats,
    )


def tokenize_split_to_shards(
    split: Dataset,
    text_field: str,
    tokenizer: Any,
    vocab_size: int,
    bos_token_id: int,
    eos_token_id: int,
    split_dir: Path,
    tokenize_workers: int | None = None,
    max_in_flight: int | None = None,
    batch_size: int = 1_000,
    shard_size_mb: int = 512,
    flush_buffer_mb: int = 64,
) -> SplitWriteResult:
    """Tokenize one split and stream-write sharded .bin files in dataset order."""
    if tokenize_workers is None:
        tokenize_workers = min(8, os.cpu_count() or 1)
    tokenize_workers = max(1, tokenize_workers)

    dtype = np.dtype(np.uint16 if vocab_size < 65536 else np.uint32)
    itemsize = dtype.itemsize
    tokens_per_shard = max(1, (shard_size_mb * 1024 * 1024) // itemsize)
    flush_tokens = max(1, (flush_buffer_mb * 1024 * 1024) // itemsize)
    prepare_split_dir(split_dir)

    writer = ShardedBinWriter(
        split_dir=split_dir,
        dtype=dtype,
        tokens_per_shard=tokens_per_shard,
        flush_tokens=flush_tokens,
    )
    token_count = 0
    example_count = 0
    length_stats = init_length_stats()

    pbar = tqdm(
        total=None,
        desc=f"Tokenizing + writing {split_dir.name} shards (workers={tokenize_workers})",
        unit="tok",
        unit_scale=True,
        dynamic_ncols=True,
    )

    if max_in_flight is None:
        max_in_flight = tokenize_workers * 2
    max_in_flight = max(1, max_in_flight)
    next_submit_idx = 0
    next_write_idx = 0
    in_flight: dict[concurrent.futures.Future[BatchTokenizeResult], int] = {}
    ready: dict[int, BatchTokenizeResult] = {}

    def drain_ready_batches() -> None:
        nonlocal next_write_idx, token_count, example_count
        while next_write_idx in ready:
            batch_result = ready.pop(next_write_idx)
            for arr in batch_result.arrays:
                writer.add(arr)

            token_count += batch_result.token_count
            example_count += batch_result.example_count
            if batch_result.example_count > 0:
                merge_length_stats(length_stats, batch_result.length_stats_raw)
            pbar.update(batch_result.token_count)
            next_write_idx += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=tokenize_workers) as executor:
        for batch in split.iter(batch_size=batch_size):
            future = executor.submit(
                _tokenize_text_batch,
                batch[text_field],
                tokenizer,
                bos_token_id,
                eos_token_id,
                dtype,
            )
            in_flight[future] = next_submit_idx
            next_submit_idx += 1

            if len(in_flight) >= max_in_flight:
                done, _ = concurrent.futures.wait(
                    in_flight.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                )
                for finished in done:
                    batch_idx = in_flight.pop(finished)
                    ready[batch_idx] = finished.result()
                drain_ready_batches()

        while in_flight:
            done, _ = concurrent.futures.wait(
                in_flight.keys(), return_when=concurrent.futures.FIRST_COMPLETED
            )
            for finished in done:
                batch_idx = in_flight.pop(finished)
                ready[batch_idx] = finished.result()
            drain_ready_batches()

    pbar.close()
    stats = finalize_length_stats(length_stats, example_count)
    shards = writer.close()
    return SplitWriteResult(
        token_count=token_count,
        example_count=example_count,
        dtype=dtype,
        length_stats=stats,
        shards=shards,
    )
