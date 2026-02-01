"""BPE tokenizer training."""

from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def train_bpe_tokenizer(
    text_iterator: Iterator[str],
    vocab_size: int,
    special_tokens: list[str],
    output_dir: str | Path,
) -> Tokenizer:
    """
    Train a BPE tokenizer on text data.

    Args:
        text_iterator: Iterator yielding text strings
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens (pad, eos, bos, unk)
        output_dir: Directory to save the trained tokenizer

    Returns:
        Trained tokenizer
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token=special_tokens[3]))
    tokenizer.pre_tokenizer = Whitespace()

    # Configure trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Train
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)

    # Save
    tokenizer_path = output_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    return tokenizer
