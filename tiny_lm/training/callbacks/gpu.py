"""GPU stats logging callback."""

from __future__ import annotations

import shutil
import subprocess

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_only


class GpuStatsMonitor(pl.Callback):
    """
    Log GPU memory usage and temperature.

    Currently tuned for single-GPU: all stats are logged from rank 0 only.
    """

    def __init__(self, log_every_n_steps: int = 1) -> None:
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be positive")
        self.log_every_n_steps = log_every_n_steps
        self._nvidia_smi = shutil.which("nvidia-smi")
        self._last_logged_step = -1

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = trainer.global_step
        if step % self.log_every_n_steps != 0 or step == self._last_logged_step:
            return
        self._last_logged_step = step

        if torch.cuda.is_available() and pl_module.device.type == "cuda":
            device = pl_module.device
            mem_mb = torch.cuda.memory_allocated(device) / (1024**2)
            max_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            pl_module.log(
                "gpu_mem_mb",
                mem_mb,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )
            pl_module.log(
                "gpu_mem_max_mb",
                max_mem_mb,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

            if self._nvidia_smi is not None:
                try:
                    device_index = torch.cuda.current_device()
                    output = subprocess.check_output(
                        [
                            self._nvidia_smi,
                            "--query-gpu=temperature.gpu",
                            "--format=csv,noheader,nounits",
                            "-i",
                            str(device_index),
                        ],
                        text=True,
                        timeout=1.0,
                    ).strip()
                    if output:
                        pl_module.log(
                            "gpu_temp_c",
                            float(output),
                            on_step=True,
                            on_epoch=False,
                            prog_bar=False,
                            logger=True,
                        )
                except (subprocess.SubprocessError, ValueError):
                    pass
