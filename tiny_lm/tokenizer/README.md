# Tokenizer

Minimal tokenizer setup inspired by Karpathy's nanochat:
https://github.com/karpathy/nanochat

We train with the Rust BPE implementation from `rustbpe` and export a tiktoken
encoding for fast, simple inference. This keeps the runtime surface small and
avoids HuggingFace tokenizer complexity.

Reference: https://github.com/karpathy/rustbpe

The regex pre-tokenization follows the GPT-4 style pattern used in nanochat.
It preserves newlines and keeps word boundaries to avoid memory blowups.

Output: `tokenizer.pkl` (tiktoken encoding)
Metadata: `metadata.txt` is generated alongside the tokenizer for quick inspection (special tokens, vocab size, and config notes).
