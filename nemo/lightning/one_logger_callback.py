"""
OneLogger callback for NeMo training.

This module provides a callback that integrates OneLogger telemetry with NeMo training.
"""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
from one_logger_utils.core import OneLogger
from pytorch_lightning.callbacks import Callback


class OneLoggerNeMoCallback(Callback):
    """NeMo callback that integrates with OneLogger for tracking metrics.

    This callback implements NeMo's callback group API and internally
    uses OneLogger's core functionality to track metrics.
    """

    def __init__(self, callback_config: Optional[Dict[str, Any]] = None, set_barrier: bool = False):
        """Initialize the OneLogger NeMo callback.

        Args:
            callback_config: Configuration dictionary with metadata from
                MetaInfoManager(cfg).get_metadata()
            set_barrier: Whether to use barriers for synchronization
        """
        super().__init__()

        # Create a copy of the configuration to avoid modifying the original
        if callback_config is not None:
            # Map new metadata keys to old variable names expected by OneLogger
            converted_config = callback_config.copy()

            # Mapping of new names to old names
            name_mapping = {
                "app_name": "one_logger_project",
                "log_every_n_iterations": "log_every_n_train_iterations",
                "perf_version_tag": "app_tag_run_version",
                "workload_type": "app_run_type",
                "perf_tag": "app_tag",
                "session_tag": "app_tag_run_name",
            }

            # Convert names in the config
            for new_name, old_name in name_mapping.items():
                if new_name in converted_config:
                    converted_config[old_name] = converted_config.pop(new_name)
        else:
            converted_config = None

        # Create OneLogger instance with the converted metadata
        self.one_logger = OneLogger(callback_config=converted_config, set_barrier=set_barrier)
        self.world_size = callback_config.get("world_size", 1) if callback_config else 1
        self.rank = callback_config.get("rank", 0) if callback_config else 0

    def __getattr__(self, name: str) -> Any:
        """Automatically forward any undefined method calls to the underlying OneLogger instance.

        This eliminates the need for manually writing pass-through methods for each OneLogger API.
        Only methods that need custom logic (like those interacting with the trainer) need to be
        explicitly defined in this class.

        Args:
            name: The name of the method being called

        Returns:
            The method from the underlying OneLogger instance

        Raises:
            AttributeError: If the method is not found in the OneLogger instance
        """
        # Check if the method exists on the OneLogger instance
        if hasattr(self.one_logger, name):
            return getattr(self.one_logger, name)

        # If not found, raise AttributeError as normal
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when training begins.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
        """
        # Extract necessary information from the trainer
        current_step = trainer.global_step
        max_steps = trainer.max_steps if hasattr(trainer, 'max_steps') else 0

        # Call OneLogger's on_train_start with the extracted information
        self.one_logger.on_train_start(train_iterations_start=current_step, train_iterations_target=max_steps)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when training ends.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
        """
        self.one_logger.on_train_end(train_iterations_end=trainer.global_step)

    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int
    ) -> None:
        """Called when a training batch begins.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
            batch: The current batch of data
            batch_idx: The index of the current batch
        """
        self.one_logger.on_train_batch_start(train_iterations=trainer.global_step)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called when a training batch ends.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
            outputs: The outputs from the training step
            batch: The current batch of data
            batch_idx: The index of the current batch
        """
        self.one_logger.on_train_batch_end()

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when validation begins.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
        """
        self.one_logger.on_validation_start(val_iterations=trainer.global_step)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when validation ends.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
        """
        self.one_logger.on_validation_end(val_iterations=trainer.global_step)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when testing begins.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
        """
        self.one_logger.on_test_start(test_iterations=trainer.global_step)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when testing ends.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
        """
        self.one_logger.on_test_end(test_iterations=trainer.global_step)
