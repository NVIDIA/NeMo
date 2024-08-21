import copy
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import nemo_run as run
import yaml
from nemo_run.core.serialization.yaml import YamlSerializer
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger

from nemo.lightning.pytorch.callbacks import NsysCallback, PreemptionCallback


def _merge_callbacks(partial: run.Partial, callbacks: list[run.Config[Callback]]):
    if hasattr(partial, "trainer"):
        if hasattr(partial.trainer, "callbacks"):
            for callback in callbacks:
                if callback not in partial.trainer.callbacks:
                    partial.trainer.callbacks.append(callback)
        else:
            partial.trainer.callbacks = copy.deepcopy(callbacks)


@dataclass(kw_only=True)
class PreemptionPlugin(run.Plugin):
    callbacks: list[run.Config[Callback]] = field(default_factory=lambda: [run.Config(PreemptionCallback)])

    def setup(self, task: run.Partial | run.Script, executor: run.Executor):
        if isinstance(task, run.Script):
            logging.warning(
                f"The {self.__class__.__name__} will have no effect on the task as it's an instance of run.Script"
            )
            return

        if isinstance(executor, run.SlurmExecutor):
            # Sends a SIGTERM 5 minutes before hitting time limit
            executor.signal = "TERM@300"

        _merge_callbacks(task, callbacks=self.callbacks)


@dataclass(kw_only=True)
class NsysPlugin(run.Plugin):
    start_step: int
    end_step: int
    ranks: Optional[list[int]] = None
    nsys_trace: Optional[list[str]] = None

    def setup(self, task: run.Partial | run.Script, executor: run.Executor):
        if isinstance(task, run.Partial):
            nsys_callback = run.Config(
                NsysCallback,
                start_step=self.start_step,
                end_step=self.end_step,
                ranks=self.ranks or [0],
            )
            callbacks: list[run.Config[Callback]] = [nsys_callback]  # type: ignore
            _merge_callbacks(task, callbacks=callbacks)

        launcher = executor.get_launcher()
        launcher.nsys_profile = True
        launcher.nsys_trace = self.nsys_trace or ["nvtx", "cuda"]


@dataclass(kw_only=True)
class WandbPlugin(run.Plugin):
    name: str
    logger_fn: Callable[..., run.Config[WandbLogger]]
    log_task_config: bool = True

    def setup(self, task: run.Partial | run.Script, executor: run.Executor):
        if isinstance(task, run.Script):
            logging.warning(
                f"The {self.__class__.__name__} will have no effect on the task as it's an instance of run.Script"
            )
            return

        if "WANDB_API_KEY" in os.environ:
            executor.env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

            if hasattr(task, "log") and hasattr(task.log, "wandb"):
                task.log.wandb = self.logger_fn(name=self.name)
                if self.log_task_config:
                    partial_config = yaml.safe_load(YamlSerializer().serialize(task))
                    partial_config["experiment"] = {
                        "id": self.experiment_id,
                        "task_name": self.name,
                        "executor": executor.info(),
                        "remote_directory": (
                            os.path.join(executor.tunnel.job_dir, Path(executor.job_dir).name)
                            if isinstance(executor, run.SlurmExecutor)
                            else None
                        ),
                        "local_directory": executor.job_dir,
                    }
                    task.log.wandb.config = partial_config
