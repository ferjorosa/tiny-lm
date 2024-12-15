import yaml
from easydict import EasyDict
import pytorch_lightning as pl
from pytorch_lightning import loggers

from src.data.lm_data_module import LanguageModelDataModule
from src.model.lm_module import LanguageModelModule
from src.callbacks.training_time_callback import TrainingTimeCallback

# Load configuration from config/data.yaml
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
    config = EasyDict(config)

random_seed = 42

# Logger
logger = loggers.CSVLogger(
    config["logs_dir"],
    name=f"{config.name}/{config.dataset_name}",
)

tokenizer = None

data_module = LanguageModelDataModule(
    dataset_name=config.dataset_name,
    text_col=config.dataset_text_col,
    n_train_rows=config.dataset_length_train,
    n_val_rows=config.dataset_length_val,
    max_seq_length=config.max_seq_length,
    num_workers=config.num_workers,
    tokenizer=tokenizer,
    random_seed=random_seed
)

model_module = LanguageModelModule(
    model_name_or_path=config.base_model,
    learning_rate=config.learning_rate
)

trainer = pl.Trainer(
    devices=config.compute_devices,
    accelerator=config.compute_accelerator,
    min_epochs=1,
    max_epochs=config.num_epochs,
    precision=config.precision,
    logger=logger,
    callbacks=[
        TrainingTimeCallback(),
    ],
    # In case we want to work with a dataset that does not have validation
    limit_val_batches=0,
    num_sanity_val_steps=0,
)
