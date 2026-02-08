"""Progress bar callbacks."""

from __future__ import annotations

from pytorch_lightning.callbacks import TQDMProgressBar


class OptimizerStepProgressBar(TQDMProgressBar):
    """Show optimizer steps (global_step) on the train progress bar."""

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            bar.total = self.trainer.max_steps
        return bar

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.train_progress_bar is None:
            return
        self.train_progress_bar.n = trainer.global_step
        self.train_progress_bar.refresh()
