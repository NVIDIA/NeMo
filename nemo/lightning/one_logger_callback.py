# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OneLogger callback for NeMo training.

This module provides a callback that integrates OneLogger telemetry with NeMo training.
"""

import functools
import time
from typing import Any, Optional

# Centralized OneLogger import - this is the only place where nv_one_logger should be imported
try:
    import nv_one_logger.training_telemetry.api.callbacks as CB
    from nv_one_logger.training_telemetry.api.config import TrainingLoopConfig, TrainingTelemetryConfig
    from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider
    from nv_one_logger.training_telemetry.v1_adapter import V1CompatibleExporter

    HAVE_ONELOGGER = True
except (ImportError, ModuleNotFoundError):
    HAVE_ONELOGGER = False
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.core import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

from nemo.utils.meta_info_manager import (
    enable_onelogger,
    get_onelogger_init_config,
    get_onelogger_training_loop_config,
)

# Export OneLogger availability flag
__all__ = [
    'OneLoggerNeMoCallback',
    'hook_class_init_with_callbacks',
    'get_one_logger_callbacks',
    'update_one_logger_config',
]


def get_current_time_msec() -> float:
    """Get current time in milliseconds since epoch.

    Returns:
        float: Current time in milliseconds since epoch
    """
    return time.time() * 1000


def _get_onelogger_callbacks_function(name: str):
    """Get the OneLogger callback function without calling it.

    Args:
        name: The name of the callback to get
    Returns:
        The callback function or a no-op function if OneLogger is not available
    """

    def _noop(*args, **kwargs):
        pass

    if not HAVE_ONELOGGER:
        return _noop
    if hasattr(CB, name):
        return getattr(CB, name)
    else:
        raise AttributeError(f"OneLogger has no attribute {name}")


# Wrapper functions for OneLogger callbacks
def get_one_logger_callbacks(name: str, *args, **kwargs):
    """Get and call the OneLogger callbacks module if available.

    Args:
        name: The name of the callback to call
        *args: Positional arguments to pass to the callback
        **kwargs: Keyword arguments to pass to the callback
    """
    function = _get_onelogger_callbacks_function(name)
    return function(*args, **kwargs)


class OneLoggerNeMoCallback(Callback):
    """
    NeMo callback that integrates with OneLogger v2 for tracking metrics.

    This callback implements NeMo's callback group API and internally
    uses OneLogger's training telemetry functionality to track metrics.
    """

    def __init__(self):
        super().__init__()
        self._validation_batch_exists = False

    def __getattr__(self, name: str) -> Any:
        """Automatically forward any undefined method calls to the OneLogger v2 callbacks.

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
        return _get_onelogger_callbacks_function(name)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when training begins."""
        # Extract necessary information from the trainer
        current_step = trainer.global_step
        max_steps = trainer.max_steps if hasattr(trainer, 'max_steps') else 0

        get_one_logger_callbacks(
            "on_train_start", train_iterations_start=current_step, train_iterations_target_or_fn=max_steps
        )

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        """Called at the beginning of each training batch."""
        get_one_logger_callbacks("on_training_single_iteration_start")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called at the end of each training batch."""
        get_one_logger_callbacks("on_training_single_iteration_end")

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when validation begins."""
        get_one_logger_callbacks("on_validation_start")

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when validation ends."""
        if self._validation_batch_exists:
            get_one_logger_callbacks("on_validation_single_iteration_end")
            self._validation_batch_exists = False
        get_one_logger_callbacks("on_validation_end")

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called at the beginning of each validation batch."""
        if self._validation_batch_exists:
            get_one_logger_callbacks("on_validation_single_iteration_end")
        self._validation_batch_exists = True
        get_one_logger_callbacks("on_validation_single_iteration_start")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called at the end of each validation batch."""
        self._validation_batch_exists = False
        get_one_logger_callbacks("on_validation_single_iteration_end")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when training ends."""
        get_one_logger_callbacks("on_train_end")


def hook_class_init_with_callbacks(cls, start_callback: str, end_callback: str) -> None:
    """Hook a class's __init__ method with start and end callbacks.

    Args:
        cls (type): The class to hook the __init__ method of.
        start_callback (str): The name of the callback to call at the start of __init__.
        end_callback (str): The name of the callback to call at the end of __init__.
    """
    if not hasattr(cls, '__init__'):
        return

    original_init = cls.__init__

    # Check if already wrapped to avoid double wrapping
    if getattr(original_init, '_one_logger_wrapped', False):
        return

    @functools.wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        # Check if this instance has already been initialized to prevent duplicate callbacks
        # in inheritance chains
        if hasattr(self, '_one_logger_init_started'):
            # This instance is already being initialized, skip the callbacks
            return original_init(self, *args, **kwargs)

        # Mark this instance as being initialized
        self._one_logger_init_started = True

        get_one_logger_callbacks(start_callback, start_time_msec=get_current_time_msec())
        result = original_init(self, *args, **kwargs)
        get_one_logger_callbacks(end_callback, finish_time_msec=get_current_time_msec())
        return result

    # Mark as wrapped to prevent double wrapping
    wrapped_init._one_logger_wrapped = True
    cls.__init__ = wrapped_init


def _init_one_logger() -> None:
    """Initialize OneLogger with configuration from OneLoggerMetaInfoManager.

    This function initializes OneLogger by reading all configuration from OneLoggerMetaInfoManager.
    If OneLogger is already configured, this function will do nothing.

    The function reads:
    - Whether OneLogger is enabled
    - Initialization configuration
    - WandB configuration
    """
    if not HAVE_ONELOGGER:
        return

    # Check if OneLogger is enabled
    if not enable_onelogger:
        return

    # Check if OneLogger is already configured
    if TrainingTelemetryProvider.instance().one_logger_ready:
        return

    # Get initialization configuration
    app_start_time = get_current_time_msec()
    init_config = get_onelogger_init_config()
    training_telemetry_config = TrainingTelemetryConfig(**init_config)

    # Get WandB configuration
    exporter = V1CompatibleExporter(
        training_telemetry_config=training_telemetry_config,
        async_mode=False,
    )

    # Configure the provider with exporter (this automatically calls on_app_start)
    TrainingTelemetryProvider.instance().with_base_telemetry_config(training_telemetry_config).with_exporter(
        exporter.exporter
    ).configure_provider()
    get_one_logger_callbacks("on_app_start", start_time_msec=app_start_time)


def update_one_logger_config(
    trainer: Trainer,
    job_name: str,
    model: Optional[Any] = None,
) -> None:
    """Update OneLogger configuration with training loop config.

    This function updates the OneLogger configuration that was initialized early.
    It converts the provided dictionary to a TrainingLoopConfig instance.

    Args:
        config: Dict[str, Any] to construct TrainingLoopConfig
        trainer: Optional PyTorch Lightning trainer to add callback to
    """
    # Check if TrainingTelemetryProvider is already configured
    if not HAVE_ONELOGGER or not TrainingTelemetryProvider.instance().one_logger_ready:
        return

    config = get_onelogger_training_loop_config(
        trainer=trainer,
        job_name=job_name,
        model=model,
    )

    # Convert dict to TrainingLoopConfig
    training_loop_config = TrainingLoopConfig(**config)

    # Training loop specific config update
    TrainingTelemetryProvider.instance().set_training_loop_config(training_loop_config)

    # Add the OneLogger callback to the trainer if provided
    if trainer is not None:
        # Check if OneLoggerNeMoCallback is already in the trainer's callbacks
        has_onelogger_callback = any(isinstance(callback, OneLoggerNeMoCallback) for callback in trainer.callbacks)

        if not has_onelogger_callback:
            # Create the callback with metadata
            onelogger_callback = OneLoggerNeMoCallback()
            trainer.callbacks.append(onelogger_callback)


# Initialize OneLogger when this module is imported
_init_one_logger()
