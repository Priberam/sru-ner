from lightning.pytorch.callbacks import ModelCheckpoint, Callback, EarlyStopping
import os
from omegaconf import DictConfig
import torch


def get_recent_model_callback(dir_path: str):
    return ModelCheckpoint(
        dirpath=dir_path, filename="recent_model", save_on_train_epoch_end=True
    )


def get_early_stop_callback(metric_name: str, patience: int):
    return EarlyStopping(
        monitor=metric_name, mode="max", patience=patience, verbose=True
    )


def get_metric_based_checkpoint_callback(metric_name: str, dirpath: str):

    filename = f"best-{metric_name}"

    callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        monitor=f"{metric_name}",
        mode="max",
        save_top_k=1,
        save_on_train_epoch_end=False,
    )

    return callback


class ClearCUDACacheCallback(Callback):
    def __init__(self, every_n_steps=50):
        super().__init__()
        self.every_n_steps = every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.every_n_steps == 0:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            pl_module.log(
                "cuda_cleared_at_batch", batch_idx + 1, prog_bar=True, logger=False
            )
