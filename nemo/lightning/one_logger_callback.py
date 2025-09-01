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

# Centralized OneLogger import - this is the only place where nv_one_logger should be imported
try:
    import nv_one_logger.training_telemetry.api.callbacks as CB
    from nv_one_logger.api.config import OneLoggerConfig
    from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
    from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider
    from nv_one_logger.training_telemetry.integration.pytorch_lightning import (
        TimeEventCallback as OneLoggerNeMoCallback,
    )

    HAVE_ONE_LOGGER = True
except (ImportError, ModuleNotFoundError):
    HAVE_ONE_LOGGER = False
    CB = None
    TrainingTelemetryProvider = None
    OneLoggerConfig = None
    TrainingTelemetryConfig = None
    OneLoggerNeMoCallback = None

from lightning.pytorch import Trainer

from nemo.utils.meta_info_manager import (
    enable_one_logger,
    get_nemo_v1_callback_config,
    get_nemo_v2_callback_config,
    get_one_logger_init_config,
)

# Export all symbols for testing and usage
__all__ = [
    'hook_class_init_with_callbacks',
    'call_one_logger_callback',
    'update_one_logger_config',
    'get_current_time_msec',
    'init_one_logger',
    'HAVE_ONE_LOGGER',
]

_ONE_LOGGER_CALLBACK = None


def get_current_time_msec() -> float:
    """Get current time in milliseconds since epoch.

    Returns:
        float: Current time in milliseconds since epoch
    """
    return time.time() * 1000


def _call_one_logger_callbacks_function(name: str):
    """Get the OneLogger callback function without calling it.

    Args:
        name: The name of the callback to get
    Returns:
        The callback function or a no-op function if OneLogger is not available
    """

    def _noop(*args, **kwargs):
        pass

    if not HAVE_ONE_LOGGER:
        return _noop
    if hasattr(CB, name):
        return getattr(CB, name)
    else:
        raise AttributeError(f"OneLogger has no attribute {name}")


# Wrapper functions for OneLogger callbacks
def call_one_logger_callback(name: str, *args, **kwargs):
    """Call a OneLogger callback function if available.

    Args:
        name: The name of the callback to call
        *args: Positional arguments to pass to the callback
        **kwargs: Keyword arguments to pass to the callback
    """
    function = _call_one_logger_callbacks_function(name)
    return function(*args, **kwargs)


def hook_class_init_with_callbacks(cls, start_callback: str, end_callback: str) -> None:
    """Hook a class's __init__ method with start and end callbacks.

    Args:
        cls (type): The class to hook the __init__ method of.
        start_callback (str): The name of the callback to call at the start of __init__.
        end_callback (str): The name of the callback to call at the end of __init__.
    """
    if not HAVE_ONE_LOGGER or not hasattr(cls, '__init__'):
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

        call_one_logger_callback(start_callback, start_time_msec=get_current_time_msec())
        result = original_init(self, *args, **kwargs)
        call_one_logger_callback(end_callback, finish_time_msec=get_current_time_msec())
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
    - Exporters are configured via entry points
    """
    global _ONE_LOGGER_CALLBACK
    global HAVE_ONE_LOGGER

    if not HAVE_ONE_LOGGER:
        return

    # Check if OneLogger is enabled
    if not enable_one_logger:
        return

    try:
        # Check if OneLogger is already configured
        if TrainingTelemetryProvider.instance().one_logger_ready:
            return

        # Get initialization configuration
        init_config = get_one_logger_init_config()
        one_logger_config = OneLoggerConfig(**init_config)

        # Configure the provider with entry-point exporters (automatically calls on_app_start)
        TrainingTelemetryProvider.instance().with_base_config(
            one_logger_config
        ).with_export_config().configure_provider()
        _ONE_LOGGER_CALLBACK = OneLoggerNeMoCallback(TrainingTelemetryProvider.instance())
    except Exception:
        HAVE_ONE_LOGGER = False


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
    if not HAVE_ONE_LOGGER or not TrainingTelemetryProvider.instance().one_logger_ready:
        return

    if nemo_version == 'v1':
        config = get_nemo_v1_callback_config(trainer=trainer, **kwargs)
    elif nemo_version == 'v2':
        config = get_nemo_v2_callback_config(trainer=trainer, **kwargs)
    else:
        # Fall back to v1 for unknown versions
        config = get_nemo_v1_callback_config(trainer=trainer, **kwargs)

    # Convert dict to TrainingTelemetryConfig
    training_telemetry_config = TrainingTelemetryConfig(**config)

    # Training telemetry specific config update
    TrainingTelemetryProvider.instance().set_training_telemetry_config(training_telemetry_config)

    # Add the OneLogger callback to the trainer if provided
    if trainer is not None:
        # Check if OneLoggerNeMoCallback is already in the trainer's callbacks
        has_one_logger_callback = any(isinstance(callback, OneLoggerNeMoCallback) for callback in trainer.callbacks)

        if not has_one_logger_callback and _ONE_LOGGER_CALLBACK is not None:
            # Create the callback with metadata
            one_logger_callback = _ONE_LOGGER_CALLBACK
            trainer.callbacks.append(one_logger_callback)


# Initialize OneLogger when this module is imported
init_one_logger()
