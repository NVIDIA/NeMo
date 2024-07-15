from typing import List, cast

import pytorch_lightning as pl
import torch
from lightning_fabric.plugins import ClusterEnvironment
from pytorch_lightning.callbacks import Callback, TQDMProgressBar

from nemo.lightning import _strategy_lib
from nemo.lightning.pytorch.callbacks import MegatronProgressBar


class MegatronSetup(Callback):
    """Callback that implements the setup hook for setting up common elements with or without MegatronStrategy.

    With MegatronStrategy, this callback will setup the requested parallelism.
    Without MegatronStrategy (FSDP), this callback will setup the default parallelism (TP=1, PP=1, etc.)

    This callback is injected into nemo lightning trainer by default.
    """

    def setup(self, trainer, pl_module, stage):
        self._setup_nemo(trainer)
        self._setup_mcore(pl_module)
        self._setup_data_sampler(trainer)
        self._fix_progress_bar(trainer)

    def _setup_nemo(self, trainer: pl.Trainer):
        from megatron.core.model_parallel_config import ModelParallelConfig

        env = cast(ClusterEnvironment, trainer.strategy.cluster_environment)
        parallelism = getattr(trainer.strategy, "parallelism", ModelParallelConfig())
        _strategy_lib.init_parallel_ranks(env.world_size(), env.global_rank(), env.local_rank(), parallelism)

    def _setup_mcore(self, pl_module: pl.LightningModule):
        from megatron.core import parallel_state

        from nemo.utils import AppState

        if not parallel_state.model_parallel_is_initialized():
            app_state = AppState()

            if app_state.model_parallel_size is not None:
                _strategy_lib.init_model_parallel(pl_module)

    def _setup_data_sampler(self, trainer: pl.Trainer):
        datamodule = getattr(trainer, "datamodule", None)
        if hasattr(trainer.strategy, "data_sampler") and trainer.strategy.data_sampler is not None:
            datamodule.data_sampler = trainer.strategy.data_sampler
        elif hasattr(datamodule, "data_sampler"):
            trainer.strategy.data_sampler = datamodule.data_sampler
        if trainer.strategy.data_sampler is not None:
            trainer.strategy.data_sampler.setup(trainer.strategy.cluster_environment.global_rank())
            trainer.strategy.data_sampler.connect(trainer)

    def _fix_progress_bar(self, trainer: pl.Trainer) -> None:
        callbacks: List[pl.Callback] = cast(List[pl.Callback], getattr(trainer, "callbacks"))
        contains_megatron_progress, contains_progress = False, False
        for callback in callbacks:
            if isinstance(callback, MegatronProgressBar):
                contains_megatron_progress = True
            if callback.__class__ == TQDMProgressBar:
                contains_progress = True
        if not contains_megatron_progress and contains_progress:
            for callback in callbacks:
                if isinstance(callback, TQDMProgressBar):
                    callback.__class__ = MegatronProgressBar
                    break
