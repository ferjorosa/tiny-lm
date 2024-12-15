from transformers import AutoTokenizer
from tqdm import tqdm

from src.data.lm_dataset import LanguageModelDataset

def train_tokenizer(
    dataset: LanguageModelDataset,
    base_model: str,
    vocab_size: int,
    batch_size: int,
):
    def _batch_iterator():
        for i in tqdm(range(0, len(dataset), batch_size)):
            yield dataset[i: i + batch_size]

    # Load base tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(base_model, errors="ignore", use_fast=True)

    # Train new tokenizer
    tokenizer = base_tokenizer.train_new_from_iterator(
        _batch_iterator(),
        vocab_size=vocab_size
    )

    return tokenizer