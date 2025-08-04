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
import uuid
from typing import Any, Dict, Optional

# Centralized OneLogger import - this is the only place where nv_one_logger should be imported
try:
    import nv_one_logger.training_telemetry.api.callbacks as CB
    from nv_one_logger.training_telemetry.api.config import TrainingLoopConfig, TrainingTelemetryConfig
    from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider
    from nv_one_logger.api.config import OneLoggerConfig
    from nv_one_logger.training_telemetry.v1_adapter import V1CompatibleExporter
    from nv_one_logger.training_telemetry.integration.pytorch_lightning import TimeEventCallback as OneLoggerNeMoCallback

    HAVE_ONELOGGER = True
except (ImportError, ModuleNotFoundError):
    HAVE_ONELOGGER = False

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.core import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from nemo.utils.meta_info_manager import (
    get_onelogger_init_config, 
    get_nemo_v1_telemetry_config,
    get_nemo_v2_telemetry_config,
    enable_onelogger
)

# Export OneLogger availability flag
__all__ = [
    'OneLoggerNeMoCallback',
    'hook_class_init_with_callbacks',
    'get_one_logger_callbacks',
    'update_one_logger_config',
]

_ONELOGGER_CALLBACK = None


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


def init_one_logger() -> None:
    """Initialize OneLogger with configuration from OneLoggerMetaInfoManager.

    This function initializes OneLogger by reading all configuration from OneLoggerMetaInfoManager.
    If OneLogger is already configured, this function will do nothing.

    The function reads:
    - Whether OneLogger is enabled
    - Initialization configuration
    - WandB configuration
    """
    global _ONELOGGER_CALLBACK
    
    if not HAVE_ONELOGGER:
        return

    # Check if OneLogger is enabled
    if not enable_onelogger:
        return

    # Check if OneLogger is already configured
    if TrainingTelemetryProvider.instance().one_logger_ready:
        return

    # Get initialization configuration
    init_config = get_onelogger_init_config()
    one_logger_config = OneLoggerConfig(**init_config)

    # Get WandB configuration
    exporter = V1CompatibleExporter(
        one_logger_config=one_logger_config,
        async_mode=False,
    )

    # Configure the provider without exporter (this automatically calls on_app_start)
    TrainingTelemetryProvider.instance().with_base_config(one_logger_config).with_exporter(exporter.exporter).configure_provider()
    _ONELOGGER_CALLBACK = OneLoggerNeMoCallback(TrainingTelemetryProvider.instance())


def update_one_logger_config(
    nemo_version: str,
    trainer: Trainer,
    **kwargs,
) -> None:
    """Update OneLogger with the latest training telemetry configuration.

    This function updates the OneLogger configuration after initialization by
    generating and applying a TrainingTelemetryConfig based on the current training context.
    The configuration is constructed using the appropriate NeMo v1 or v2 telemetry config
    generator, depending on the provided version.

    Args:
        nemo_version: 'v1' or 'v2', selects which telemetry config generator to use.
        kwargs: Additional keyword arguments to pass to the telemetry config generator.
    """
    # Check if TrainingTelemetryProvider is already configured
    if not HAVE_ONELOGGER or not TrainingTelemetryProvider.instance().one_logger_ready:
        return

    if nemo_version == 'v1':
        config = get_nemo_v1_telemetry_config(trainer=trainer, **kwargs)
    elif nemo_version == 'v2':
        config = get_nemo_v2_telemetry_config(trainer=trainer, **kwargs)
    else:
        raise ValueError(f"Invalid NeMo version: {nemo_version}")

    # Convert dict to TrainingTelemetryConfig
    training_telemetry_config = TrainingTelemetryConfig(**config)
    
    # Training telemetry specific config update
    TrainingTelemetryProvider.instance().set_training_telemetry_config(training_telemetry_config)

    # Add the OneLogger callback to the trainer if provided
    if trainer is not None:
        # Check if OneLoggerNeMoCallback is already in the trainer's callbacks
        has_onelogger_callback = any(isinstance(callback, OneLoggerNeMoCallback) for callback in trainer.callbacks)

        if not has_onelogger_callback and _ONELOGGER_CALLBACK is not None:
            # Create the callback with metadata
            onelogger_callback = _ONELOGGER_CALLBACK
            trainer.callbacks.append(onelogger_callback)


# Initialize OneLogger when this module is imported
init_one_logger()
