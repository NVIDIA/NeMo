"""
OneLogger callback for NeMo training.

This module provides a callback that integrates OneLogger telemetry with NeMo training.
"""

import functools
import time
from typing import Any, Dict

# Centralized OneLogger import - this is the only place where nv_one_logger should be imported
try:
    import nv_one_logger.training_telemetry.api.callbacks as CB

    HAVE_ONELOGGER = True
    from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider
    from nv_one_logger.training_telemetry.v1_adapter.config_adapter import ConfigAdapter
    from nv_one_logger.training_telemetry.v1_adapter.v1_compatible_wandb_exporter import V1CompatibleWandbExporterAsync
except (ImportError, ModuleNotFoundError):
    HAVE_ONELOGGER = False
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.core import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

# Export OneLogger availability flag
__all__ = [
    'OneLoggerNeMoCallback',
    'OneLoggerTimingTracker',
    'hook_class_init_with_callbacks',
    'get_onelogger_callbacks',
    'init_one_logger',
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
    return _noop


# Wrapper functions for OneLogger callbacks
def get_onelogger_callbacks(name: str, *args, **kwargs):
    """Get and call the OneLogger callbacks module if available.

    Args:
        name: The name of the callback to call
        *args: Positional arguments to pass to the callback
        **kwargs: Keyword arguments to pass to the callback
    """
    function = _get_onelogger_callbacks_function(name)
    return function(*args, **kwargs)


class OneLoggerTimingTracker:
    """A singleton class to track timing data for OneLogger callbacks.

    This class stores timing data when OneLogger is not yet initialized and
    logs it when OneLogger becomes available.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> 'OneLoggerTimingTracker':
        """Get the singleton instance of the OneLoggerTimingTracker."""
        if cls._instance is None:
            cls._instance = OneLoggerTimingTracker()
        return cls._instance

    def __init__(self):
        """Initialize the timing tracker."""
        self._pending_events = []
        self._one_logger_available = False
        self.track_event('on_app_start')

    def track_event(self, event_type: str, current_time_ms: float = None):
        """Track an event with optional timestamp.

        Args:
            event_type: The type of event to track
            current_time_ms: Optional timestamp in milliseconds
        """
        if current_time_ms is None:
            current_time_ms = get_current_time_msec()

        event = {
            'event_type': event_type,
            'timestamp': current_time_ms,
        }

        if self._one_logger_available:
            self._log_event(event)
        else:
            self._pending_events.append(event)

    def set_one_logger_available(self, available: bool = True):
        """Set whether OneLogger is available and process pending events.

        Args:
            available: Whether OneLogger is available
        """
        self._one_logger_available = available
        if available and self._pending_events:
            for event in self._pending_events:
                self._log_event(event)
            self._pending_events.clear()

    @classmethod
    def mark_one_logger_available(cls):
        """Mark OneLogger as available for the singleton instance."""
        cls.get_instance().set_one_logger_available(True)

    def _log_event(self, event: Dict[str, Any]):
        """Log an event to OneLogger if available.

        Args:
            event: The event to log
        """
        if HAVE_ONELOGGER:
            get_onelogger_callbacks("log_event", event)


class OneLoggerNeMoCallback(Callback):
    """Callback for integrating OneLogger telemetry with NeMo training."""

    def __init__(self):
        """Initialize the OneLogger callback."""
        super().__init__()
        self._validation_batch_exists = False
        self._train_active = False

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

        get_onelogger_callbacks(
            "on_train_start", 
            train_iterations_start=current_step, 
            train_iterations_target_or_fn=max_steps
        )

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        """Called at the beginning of each training batch."""
        get_onelogger_callbacks("on_training_single_iteration_start")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called at the end of each training batch."""
        get_onelogger_callbacks("on_training_single_iteration_end")

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when validation begins."""
        get_onelogger_callbacks("on_validation_start")

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when validation ends."""
        if self._validation_batch_exists:
            get_onelogger_callbacks("on_validation_single_iteration_end")
            self._validation_batch_exists = False
        get_onelogger_callbacks("on_validation_end")

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
            get_onelogger_callbacks("on_validation_single_iteration_end")
        self._validation_batch_exists = True
        get_onelogger_callbacks("on_validation_single_iteration_start")

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
        get_onelogger_callbacks("on_validation_single_iteration_end")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when training ends."""
        if self._train_active:
            get_onelogger_callbacks("on_train_end")
            self._train_active = False


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
        if hasattr(self, '_one_logger_initialized'):
            return original_init(self, *args, **kwargs)

        # Call start callback
        get_onelogger_callbacks(start_callback)

        # Call original __init__
        result = original_init(self, *args, **kwargs)

        # Mark as initialized to prevent duplicate callbacks
        self._one_logger_initialized = True

        # Call end callback
        get_onelogger_callbacks(end_callback)

        return result

    # Mark as wrapped to prevent double wrapping
    wrapped_init._one_logger_wrapped = True
    cls.__init__ = wrapped_init


def init_one_logger(v1_config: Dict[str, Any], trainer: Trainer = None, enable_onelogger: bool = True):
    """Initialize OneLogger with v1-style configuration.

    Args:
        v1_config: V1-style configuration dictionary
        trainer: Optional PyTorch Lightning trainer
        enable_onelogger: Whether to enable OneLogger
    """
    if not enable_onelogger or not HAVE_ONELOGGER:
        return

    # Mark OneLogger as available
    OneLoggerTimingTracker.mark_one_logger_available()

    # Initialize OneLogger with v1 config
    if trainer is not None:
        # Add OneLogger callback to trainer
        callback = OneLoggerNeMoCallback()
        trainer.callbacks.append(callback)
