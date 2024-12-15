import torch
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW

class LanguageModelModule(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        learning_rate: float,
    ):
        super().__init__()
        # Load the model and tokenizer from the given model name or path
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.logits

    def common_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Shift labels for next-token prediction
        shift_labels = labels[..., 1:].contiguous()
        shift_input_ids = input_ids[..., 1:].contiguous()
        shift_attention_mask = attention_mask[..., 1:].contiguous()

        # Get logits
        logits = self(shift_input_ids, shift_attention_mask)

        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            shift_labels.view(-1)
        )

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Generate predictions
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=50  # Adjust as needed
        )

        return generated_ids

    def configure_optimizers(self):
        # AdamW optimizer
        return AdamW(self.parameters(), lr=self.learning_rate)