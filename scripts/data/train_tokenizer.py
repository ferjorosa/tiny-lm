"""Train a BPE tokenizer on a dataset."""

import argparse
import os
from pathlib import Path

from tiny_lm.dataset import load_dataset_from_config
from tiny_lm.tokenizer.config import TokenizerConfig
from tiny_lm.tokenizer.trainer_rust import train_bpe_tokenizer_rust


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tokenizers/tinystories-8k.yaml",
        help="Path to tokenizer config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = TokenizerConfig.from_yaml(config_path)
    num_threads = config.num_threads or max(1, (os.cpu_count() or 4))

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("RAYON_NUM_THREADS", str(num_threads))

    # Load dataset
    print(f"Loading dataset from {config.dataset_config}...")
    dataset, dataset_config = load_dataset_from_config(config.dataset_config)

    # Create text iterator
    def text_iterator():
        for example in dataset["train"]:
            yield example[dataset_config.text_field]

    print(f"\nTraining BPE tokenizer (vocab_size={config.vocab_size})...")

    enc = train_bpe_tokenizer_rust(
        text_iterator=text_iterator(),
        vocab_size=config.vocab_size,
        pad_token=config.special_tokens["pad"],
        eos_token=config.special_tokens["eos"],
        bos_token=config.special_tokens["bos"],
        unk_token=config.special_tokens["unk"],
        output_dir=config.output_dir,
    )
    print(f"Tokenizer saved to {config.output_dir}/tokenizer.pkl")

    # Test rust tokenizer
    print("\nTesting tokenizer:")
    test_text = "Once upon a time, there was a little girl."
    tokens = enc.encode_ordinary(test_text)
    print(f"  Input: {test_text}")
    print(f"  IDs: {tokens[:10]}...")
    print(f"  Decoded: {enc.decode(tokens)}")


if __name__ == "__main__":
    main()
