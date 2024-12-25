import yaml
import pytorch_lightning as pl
import logging
from easydict import EasyDict
from pytorch_lightning import loggers
from transformers import AutoTokenizer, Qwen2Config

from src.data.lm_data_module import LanguageModelDataModule
from src.model.lm_module import LanguageModelModule
from src.callbacks.training_time_callback import TrainingTimeCallback


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load configuration from config/data.yaml
    logger.info("Loading configuration from config.yaml")
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        config = EasyDict(config)

    random_seed = 42

    # Logger
    logger.info("Initializing logger for experiment tracking")
    experiment_logger = loggers.CSVLogger(
        config["logs_dir"],
        name=f"{config.name}/{config.dataset_name}",
    )

    # Tokenizer
    logger.info("Initializing tokenizer from tokenizer file")
    tokenizer_dir = f"tokenizer/{config.name}/{config.dataset_name}"

    # Load the tokenizer with special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        use_fast=True,
        # Ensure special tokens are loaded
        special_tokens_map_file=f"{tokenizer_dir}/special_tokens_map.json",
        tokenizer_config_file=f"{tokenizer_dir}/tokenizer_config.json"
    )


    # Data Module
    logger.info("Creating LanguageModelDataModule with provided configuration")
    data_module = LanguageModelDataModule(
        dataset_name=config.dataset_name,
        text_col=config.dataset_text_col,
        n_train_rows=config.dataset_length_train,
        n_val_rows=config.dataset_length_val,
        batch_size=config.batch_size,
        max_seq_length=config.max_seq_length,
        num_workers=config.num_workers,
        tokenizer=tokenizer,
        random_seed=random_seed
    )

    # Model Module
    logger.info("Creating LanguageModelModule with provided configuration")
    model_config = Qwen2Config(
        num_hidden_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        intermediate_size=config.hidden_size * 4,  # MLP hidden dim, following GPT-2 approach x4
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads, # if equal to the num_attention heads, the MHA if 1 then MQA, else GQA
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_position_embeddings,  # Maximum sequence length
        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
    )

    model_module = LanguageModelModule(
        model_config=model_config,
        tokenizer=tokenizer,
        learning_rate=config.learning_rate,
    )

    # Trainer
    logger.info("Setting up PyTorch Lightning Trainer with provided configuration")
    trainer = pl.Trainer(
        devices=config.compute_devices,
        accelerator=config.compute_accelerator,
        min_epochs=1,
        max_epochs=config.num_epochs,
        precision=config.precision,
        logger=experiment_logger,
        callbacks=[
            TrainingTimeCallback(),
        ],
        # In case we want to work with a dataset that does not have validation
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )

    # Start training
    logger.info("Starting model training")
    trainer.fit(model_module, data_module)

if __name__ == "__main__":
    main()