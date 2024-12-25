from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):
    """
    Iterable dataset where we know the number of samples.

    For those cases where we do not want to load a dataset
    fully in memory but we want to know the number of samples during training.
    """

    def __init__(
        self,
        data,
        data_length,
        text_col: str = "text",
    ):
        self.data_iterator = iter(data)
        self.data_length = data_length
        self.text_col = text_col

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        return next(self.data_iterator)[self.text_col]
