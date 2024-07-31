import sys
from typing import Any

import torch
from megatron.core.num_microbatches_calculator import get_num_microbatches
from pytorch_lightning.callbacks.progress import ProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing_extensions import override


## helpers from megatron core
def is_last_rank():
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)


class MegatronProgress(ProgressBar):
    """
    Callback for logging progress in Megatron. Prints status in terms of global batches rather than microbatches.
    """

    def format_string(self, prefix, metrics):
        log_string = prefix
        for metric, val in metrics.items():
            if val.is_integer():
                val = int(val)
                log_string += f' | {metric}: {val}'
            else:
                log_string += f' | {metric}: {val:.4}'
        return log_string

    def get_current_epoch_step(self, trainer) -> int:
        """
        Get the value of step within an epoch.
        """
        return max(
            trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.current.completed,
            trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.current.completed,
        )

    @override
    def on_train_epoch_start(self, trainer, *_):
        if trainer.max_steps > 0:
            # while resuming from a ckpt use trainer.max_steps as the total for progress bar as trainer.num_training_batches
            # is truncated to max_steps - step being resumed at
            self.total = trainer.max_steps
        else:
            self.total = trainer.num_training_batches

    @override
    def on_train_batch_end(self, trainer, pl_module, *_, **__):
        n = self.get_current_epoch_step(trainer)
        metrics = self.get_metrics(trainer, pl_module)
        prefix = f"Epoch {trainer.current_epoch}, iteration {n}/{self.total}:"
        log_string = self.format_string(prefix, metrics)
        print_rank_last(log_string)

    @override
    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return
        self.total_validation_steps = int(self.total_val_batches_current_dataloader / get_num_microbatches())

    @override
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        n = int((batch_idx + 1) / get_num_microbatches())
        print_rank_last(f"Validation: iteration {n}/{self.total_validation_steps}")

    @override
    def on_test_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return
        self.total_test_steps = int(self.total_test_batches_current_dataloader / get_num_microbatches())

    @override
    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        n = int((batch_idx + 1) / get_num_microbatches())
        print_rank_last(f"Test: iteration {n}/{self.total_validation_steps}")
