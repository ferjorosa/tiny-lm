import time
import pytorch_lightning as pl
from pathlib import Path


class TrainingTimeCallback(pl.Callback):
    def __init__(self):
        super().__init__()

        self.training_times = []
        self.file_name = "training_times.txt"

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        # Calculate the training time for the epoch
        training_time = time.time() - self.start_time

        # Convert the training time to hours, minutes, and seconds
        training_time_in_hms = time.strftime(
            "%H:%M:%S", time.gmtime(training_time)
        )

        # Store the training time for the epoch
        self.training_times.append(training_time_in_hms)

        # Update the start time for the next epoch
        self.start_time = time.time()

    def on_train_end(self, trainer, pl_module):
        # Calculate the total training time
        total_training_time = sum(
            [
                time.strptime(t, "%H:%M:%S").tm_hour * 3600
                + time.strptime(t, "%H:%M:%S").tm_min * 60
                + time.strptime(t, "%H:%M:%S").tm_sec
                for t in self.training_times
            ]
        )

        # Convert the total training time to hours, minutes, and seconds
        total_training_time_in_hms = time.strftime(
            "%H:%M:%S", time.gmtime(total_training_time)
        )

        # Get the checkpoint directory using pathlib
        checkpoint_dir = Path(trainer.checkpoint_callback.dirpath)
        save_dir = checkpoint_dir.parent

        # Construct the full path for the times file
        file_path = save_dir / self.file_name

        # Save the training times to a file
        with open(file_path, "w") as f:
            for epoch_index, training_time in enumerate(self.training_times):
                f.write(f"epoch {epoch_index}: {training_time}\n")

            f.write(f"total: {total_training_time_in_hms}\n")
