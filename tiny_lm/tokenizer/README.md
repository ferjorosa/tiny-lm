# Tokenizer Backends

Two BPE tokenizer training backends are available:

## HuggingFace (`trainer_hf.py`)

**Pros:**
- Pure Python, easier to debug
- Works out of the box with `tokenizers` package
- Parallelism via `enable_parallelism()`

**Cons:**
- Can be slower for large datasets
- Higher memory usage during training
- Inference via HuggingFace tokenizers

**Usage:**
This backend is not wired into the training script right now.

**Output:** `tokenizer.json`

## RustBPE + tiktoken (`trainer_rust.py`) - Current Default

**Pros:**
- Much faster training (native Rust implementation)
- Lower memory usage
- Faster inference via tiktoken
- Production-tested in nanochat

**Cons:**
- Requires additional dependencies: `rustbpe`, `tiktoken`
- Less debugging visibility

**Usage:**
```bash
# Install dependencies first
uv pip install rustbpe tiktoken

# Train with rust backend
uv run python scripts/tokenizer/train_tokenizer.py
```

**Output:** `tokenizer.pkl` (tiktoken encoding)

## Pattern

Both use the same GPT-4 style regex pattern that preserves newlines:
- Splits text on word boundaries (prevents memory explosion)
- Keeps `\n` in tokens for text generation
- Uses `\p{N}{1,2}` for numbers (optimized for smaller vocabs)

## Recommendation

We currently train with **RustBPE** by default. If you want to use the
HuggingFace backend later, we can wire it back in.
