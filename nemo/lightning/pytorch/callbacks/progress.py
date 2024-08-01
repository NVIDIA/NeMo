from collections import defaultdict
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

    _train_description = "Training"
    _validation_description = "Validation"
    _test_description = "Testing"
    _log_interval = 1
    # most recent "global_step" will be logged
    # rather than averaging over last log_interval steps
    _skip_accumulate_metrics = ["global_step"]
    total_metrics_dict = defaultdict(lambda: 0.)

    def __init__(
        self,
        log_interval: int = 1,
        skip_accumulate_metrics: list[str] = ["global_step"],
    ):
        self._log_interval = log_interval
        self._skip_accumulate_metrics = skip_accumulate_metrics

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

    @property
    def average_metrics_dict(self):
        average_dict = {}
        for key in self.total_metrics_dict:
            if key in self.skip_accumulate_metrics:
                average_dict[key] = self.total_metrics_dict[key]
            else:
                average_dict[key] = self.total_metrics_dict[key] / self.log_interval
        return average_dict

    @property
    def train_description(self):
        return self._train_description

    @property
    def validation_description(self):
        return self._validation_description

    @property
    def test_description(self):
        return self._test_description

    @property
    def log_interval(self):
        return self._log_interval

    @log_interval.setter
    def log_interval(self, val):
        self._log_interval = val

    @property
    def skip_accumulate_metrics(self):
        return self._skip_accumulate_metrics

    @skip_accumulate_metrics.setter
    def skip_accumulate_metrics(self, val):
        self._skip_accumulate_metrics = val

    @override
    def on_sanity_check_start(self, *_: Any) -> None:
        self._validation_description = "Sanity checking " + self.validation_description

    @override
    def on_sanity_check_end(self, *_: Any) -> None:
        self._validation_description = "Validation"

    @override
    def on_train_epoch_start(self, trainer, *_):
        if trainer.max_steps > 0:
            # while resuming from a ckpt use trainer.max_steps as the total for progress bar as trainer.num_training_batches
            # is truncated to max_steps - step being resumed at
            self.total = trainer.max_steps
        else:
            self.total = trainer.num_training_batches

    ## TODO: handle nan losses!
    @override
    def on_train_batch_end(self, trainer, pl_module, *_, **__):
        n = self.get_current_epoch_step(trainer)
        metrics = self.get_metrics(trainer, pl_module)
        for key in metrics:
            if key in self.skip_accumulate_metrics:
                self.total_metrics_dict[key] = metrics[key]
            else:
                self.total_metrics_dict[key] += metrics[key]

        if n % self.log_interval == 0:
            prefix = self.train_description + f" epoch {trainer.current_epoch}, iteration {n}/{self.total}:"
            log_string = self.format_string(prefix, self.average_metrics_dict)
            print_rank_last(log_string)

            self.total_metrics_dict = defaultdict(lambda: 0.0)

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
        print_rank_last(self.validation_description + f": iteration {n}/{self.total_validation_steps}")

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
        print_rank_last(self.test_description + f": iteration {n}/{self.total_validation_steps}")
