import yaml
from easydict import EasyDict
from datasets import load_dataset

from src.data.lm_dataset import LanguageModelDataset
from src.tokenizer.utils import train_tokenizer

# Load configuration from config/data.yaml
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
    config = EasyDict(config)

# Load the dataset
dataset = load_dataset(config.dataset_name, streaming=True, trust_remote_code=True)

# Create the LanguageModelDataset
lm_dataset = LanguageModelDataset(
    data=dataset['train'],
    data_length=config.dataset_length_train,
    text_col=config.dataset_text_col
)

# Train the tokenizer
tokenizer = train_tokenizer(
    dataset=lm_dataset,
    base_model="Qwen/Qwen2.5-0.5B",
    vocab_size=config.vocab_size,
    batch_size=config.tokenizer_batch_size
)

tokenizer.save_pretrained(f"tokenizer/{config.name}/{config.dataset_name}")