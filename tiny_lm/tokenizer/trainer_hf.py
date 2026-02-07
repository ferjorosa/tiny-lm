"""BPE tokenizer training (HuggingFace tokenizers)."""

from pathlib import Path
from typing import Iterator

from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


def train_bpe_tokenizer(
    text_iterator: Iterator[str],
    vocab_size: int,
    pad_token: str,
    eos_token: str,
    bos_token: str,
    unk_token: str,
    output_dir: str | Path,
    length: int | None = None,
) -> Tokenizer:
    """
    Train a BPE tokenizer on text data.

    Args:
        text_iterator: Iterator yielding text strings
        vocab_size: Target vocabulary size
        pad_token: Padding token string
        eos_token: End-of-sequence token string
        bos_token: Beginning-of-sequence token string
        unk_token: Unknown token string
        output_dir: Directory to save the trained tokenizer
        length: Number of examples in iterator (for memory-efficient training)

    Returns:
        Trained tokenizer
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer with byte-level BPE
    tokenizer = Tokenizer(BPE(unk_token=unk_token, byte_fallback=True))

    # Pre-tokenizer: GPT-4 style pattern that preserves newlines
    # Splits on word boundaries to prevent memory explosion while keeping \n in tokens
    # Pattern matches: contractions, letters, digits, punctuation, whitespace+newlines
    gpt4_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                pattern=Regex(gpt4_pattern), behavior="isolated", invert=False
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()

    # Enable parallelism
    tokenizer.enable_parallelism()

    # Configure trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[pad_token, eos_token, bos_token, unk_token],
        show_progress=True,
    )

    # Train with length parameter for memory-efficient processing
    tokenizer.train_from_iterator(text_iterator, trainer=trainer, length=length)

    bos_id = tokenizer.token_to_id(bos_token)
    eos_id = tokenizer.token_to_id(eos_token)
    if bos_id is None or eos_id is None:
        raise ValueError("BOS/EOS tokens must exist in the tokenizer vocab")
    tokenizer.post_processor = TemplateProcessing(
        single=f"{bos_token} $A {eos_token}",
        special_tokens=[(bos_token, bos_id), (eos_token, eos_id)],
    )

    # Save
    tokenizer_path = output_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    return tokenizer
