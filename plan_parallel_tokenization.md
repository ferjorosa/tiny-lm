# Parallel Tokenization Plan

- Goal: make `scripts/data/tokenize_data_sharded.py` write in parallel without changing global token order.
- Partitioning: split each split (`train`, `val`) into contiguous example ranges (`N` partitions).
- Worker task: each worker tokenizes its own range and writes local shard files (for example `tmp/p{idx}/part-xxxxx.bin`) with buffered writes.
- Order guarantee: keep partition index order fixed; no shuffling.
- Merge step: concatenate/rename partition shard outputs in partition order into final `train/part-xxxxx.bin` and `val/part-xxxxx.bin`.
- Metadata: aggregate per-part token counts and stats into final `metadata.json` and `manifest.json`.
- CLI: add `--write-workers` (default maybe 4) and `--partitions` (or derive from workers).
- Validation: compare total token counts and first/last token slices vs current single-writer output on a small run.
- Ops: run with `HF_DATASETS_CACHE` on `ssd2` to avoid filling `/`.
