# Tokenized Storage Bottleneck

## Issue

`scripts/data/tokenize_data.py` tokenizes with multiprocessing, but then writes `train.bin` and `val.bin` in a single Python loop with many small writes.  
Result: tokenization is fast, but final storage can become the bottleneck.

## Why this happens

- `split.map(..., num_proc=...)` parallelizes tokenization only.
- Final `.bin` writing is mostly single-threaded.
- Many per-example `tofile` calls increase overhead.
- Intermediate cache materialization can add extra write cost before final `.bin`.

## Possible approaches

1. **Chunked single-file write (low effort)**  
   Buffer many tokenized examples and write larger blocks to reduce syscall overhead.

2. **Sharded output files (medium effort)**  
   Write multiple `train-*.bin` shards in parallel, then optionally merge later.

3. **Streaming pipeline (higher effort)**  
   Avoid large intermediate materialization and stream tokenized batches directly to output files.

4. **I/O placement tuning (low effort)**  
   Put cache/output on fastest disk and reduce duplicated writes.

## Recommendation

Start with **chunked single-file writes** first (simple and likely high impact).  
If still slow, move to **parallel sharded output**.
