"""Train a BPE tokenizer on a dataset."""

import sys
from pathlib import Path

from tiny_lm.data import load_dataset_from_config
from tiny_lm.tokenizer import train_bpe_tokenizer
from tiny_lm.tokenizer.config import TokenizerConfig


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/tokenizer/train_tokenizer.py <config_path>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = TokenizerConfig.from_yaml(config_path)

    # Load dataset
    print(f"Loading dataset from {config.dataset_config}...")
    dataset, dataset_config = load_dataset_from_config(config.dataset_config)

    # Create text iterator
    def text_iterator():
        for example in dataset["train"]:
            yield example[dataset_config.text_field]

    # Train tokenizer
    special_tokens = [
        config.special_tokens["pad"],
        config.special_tokens["eos"],
        config.special_tokens["bos"],
        config.special_tokens["unk"],
    ]

    print(f"\nTraining BPE tokenizer (vocab_size={config.vocab_size})...")
    tokenizer = train_bpe_tokenizer(
        text_iterator=text_iterator(),
        vocab_size=config.vocab_size,
        special_tokens=special_tokens,
        output_dir=config.output_dir,
    )
    print(f"Tokenizer saved to {config.output_dir}/tokenizer.json")

    # Test tokenizer
    print("\nTesting tokenizer:")
    test_text = "Once upon a time, there was a little girl."
    tokens = tokenizer.encode(test_text)
    print(f"  Input: {test_text}")
    print(f"  Tokens: {tokens.tokens[:10]}...")
    print(f"  IDs: {tokens.ids[:10]}...")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("configs/tokenizers/tinystories-8k.yaml")
    main()
