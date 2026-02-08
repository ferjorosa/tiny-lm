"""BPE tokenizer training using rustbpe + tiktoken (from Karpathy's nanochat)."""

import pickle
import rustbpe
import tiktoken
from pathlib import Path
from typing import Iterator

# GPT-4 style split pattern that preserves newlines
# Note: Uses \p{N}{1,2} instead of \p{N}{1,3} for smaller vocab sizes
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def train_bpe_tokenizer_rust(
    text_iterator: Iterator[str],
    vocab_size: int,
    pad_token: str,
    eos_token: str,
    bos_token: str,
    unk_token: str,
    output_dir: str | Path,
) -> tiktoken.Encoding:
    """
    Train a BPE tokenizer using rustbpe and create tiktoken encoding.
    This is Karpathy's approach for efficient training and inference.

    Args:
        text_iterator: Iterator yielding text strings
        vocab_size: Target vocabulary size
        pad_token: Padding token string
        eos_token: End-of-sequence token string
        bos_token: Beginning-of-sequence token string
        unk_token: Unknown token string
        output_dir: Directory to save the trained tokenizer

    Returns:
        tiktoken.Encoding for inference
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    special_tokens_list = [pad_token, eos_token, bos_token, unk_token]

    # Train using rustbpe
    print("Training with rustbpe...")
    tokenizer = rustbpe.Tokenizer()  # type: ignore[attr-defined] - rustbpe exposes this at runtime
    vocab_size_no_special = vocab_size - len(special_tokens_list)
    if vocab_size_no_special < 256:
        raise ValueError(
            f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        )

    tokenizer.train_from_iterator(
        text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN
    )

    # Construct tiktoken encoding for inference
    print("Building tiktoken encoding...")
    pattern = tokenizer.get_pattern()
    mergeable_ranks_list = tokenizer.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}

    # Add special tokens after the base vocabulary
    tokens_offset = len(mergeable_ranks)
    special_tokens = {
        name: tokens_offset + i for i, name in enumerate(special_tokens_list)
    }

    enc = tiktoken.Encoding(
        name="tiny_lm_bpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    # Save tiktoken encoding
    pickle_path = output_path / "tokenizer.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(enc, f)
    print(f"Saved tiktoken encoding to {pickle_path}")

    # Also save a metadata file for easier inspection
    metadata_path = output_path / "metadata.txt"
    with open(metadata_path, "w") as f:
        f.write("Tokenizer: rustbpe + tiktoken\n")
        f.write(f"Vocab size: {enc.n_vocab}\n")
        f.write(f"Special tokens: {special_tokens_list}\n")
        f.write(f"Split pattern: {pattern}\n")
    print(f"Saved metadata to {metadata_path}")

    return enc
