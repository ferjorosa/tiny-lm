import pytorch_lightning as pl

from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerFast

class LanguageModelDataModule(pl.LightningDataModule):

    def __init__(
            self,
            dataset_name: str,
            text_col: str,
            n_train_rows: int,
            n_val_rows: int,
            batch_size: int,
            max_seq_length: int,
            num_workers: int,
            tokenizer: PreTrainedTokenizerFast,
            random_seed: int = 42,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.text_col = text_col
        self.n_train_rows = n_train_rows
        self.n_val_rows = n_val_rows
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.random_seed = random_seed

    def setup(self, stage: str):
        # Load dataset in streaming mode
        ds = load_dataset(
            self.dataset_name,
            streaming=True,
            trust_remote_code=True
        )

        # Create dataset
        self.train_ds = self._create_dataset(
            ds=ds,
            split="train",
            n_rows=self.n_train_rows,
        )

        # Try to set validation dataset if exists
        try:
            self.val_ds = self._create_dataset(
                ds=ds,
                split="validation",
                n_rows=self.n_train_rows
            )
        except Exception:
            self.val_ds = None

        # Tokenizer
        # TODO: In reality, we would initialize the tokenizer here


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            collate_fn=self._collate_batch,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            collate_fn=self._collate_batch,
            num_workers=self.num_workers,
        )

    def _create_dataset(self, ds, split, n_rows):
        # Download and load N rows
        rows = list(ds[split].take(n_rows))
        return Dataset.from_list(rows)

    def _collate_batch(self, batch):
        # Extract text from batch
        batch_text = [item[self.text_col] for item in batch]

        # Tokenize texts
        batch_tokenized = self.tokenizer(
            batch_text,
            truncation=True,
            padding="longest",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        # Prepare labels by shifting input_ids
        labels = batch_tokenized["input_ids"].clone()
        labels[:, :-1] = batch_tokenized["input_ids"][:, 1:]
        labels[:, -1] = self.tokenizer.pad_token_id

        # Add labels to the returned dictionary
        batch_tokenized["labels"] = labels

        return batch_tokenized