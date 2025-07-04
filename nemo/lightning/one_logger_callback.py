"""
OneLogger callback for NeMo training.

This module provides a callback that integrates OneLogger telemetry with NeMo training.
"""

import functools
import logging
from typing import Any, Dict, List, Optional, Type

import nv_one_logger.training_telemetry.api.callbacks as CB
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core import LightningModule
from pytorch_lightning.plugins.io import AsyncCheckpointIO
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT


class OneLoggerNeMoCallback(Callback):
    """
    NeMo callback that integrates with OneLogger v2 for tracking metrics.

    This callback implements NeMo's callback group API and internally
    uses OneLogger's training telemetry functionality to track metrics.
    """

    def __init__(
        self,
        callback_config: Optional[Dict[str, Any]] = None,
        log_interval: int = 1,
        async_io_checkpoint_classes: List[Type[Any]] | None = None,
    ):
        """
        Initialize the OneLogger NeMo callback.

        Args:
            callback_config (dict): Configuration dictionary with metadata
                from MetaInfoManager(cfg).get_metadata()
            log_interval (int): How often to log metrics
            async_io_checkpoint_classes (List[Type]): Additional classes to identify as async checkpoints
        """
        super().__init__()
        self.log_interval = log_interval
        self.async_io_checkpoint_classes = async_io_checkpoint_classes or []
        self.state = {
            "is_async_checkpoint": None,
        }

        # Extract configuration values
        if callback_config is not None:
            self.app_name = callback_config.get("app_name", "")
            self.perf_tag = callback_config.get("perf_tag", "")
            self.session_tag = callback_config.get("session_tag", "")
            self.global_batch_size = callback_config.get("global_batch_size", 0)
        else:
            self.app_name = ""
            self.perf_tag = ""
            self.session_tag = ""
            self.global_batch_size = 0

    def __getattr__(self, name: str) -> Any:
        """Automatically forward any undefined method calls to the OneLogger v2 callbacks mainly for non-trainer methods.

        This eliminates the need for manually writing pass-through methods for each OneLogger API.
        Only methods that need custom logic (like those interacting with the trainer) need to be
        explicitly defined in this class.

        Args:
            name: The name of the method being called
        Returns:
            The method from the OneLogger v2 callbacks
        Raises:
            AttributeError: If the method is not found in the OneLogger callbacks
        """
        # Check if the method exists in the OneLogger callbacks module
        if hasattr(CB, name):
            # Get the original method
            original_method = getattr(CB, name)

            # Create a wrapper that adds rank_zero_only decorator
            @functools.wraps(original_method)
            def wrapper(*args, **kwargs):
                return rank_zero_only(original_method)(*args, **kwargs)

            return wrapper

        # If not found, raise AttributeError as normal
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when training begins."""
        # Extract necessary information from the trainer
        current_step = trainer.global_step
        max_steps = trainer.max_steps if hasattr(trainer, 'max_steps') else 0

        CB.on_train_start(train_iterations_start=current_step, train_iterations_target_or_fn=max_steps)

    @rank_zero_only
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        CB.on_train_end()

    @rank_zero_only
    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        CB.on_training_single_iteration_start()

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        CB.on_training_single_iteration_end()

    @rank_zero_only
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        CB.on_validation_start()

    @rank_zero_only
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        CB.on_validation_end()

    @rank_zero_only
    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        CB.on_validation_single_iteration_start()

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        CB.on_validation_single_iteration_end()
