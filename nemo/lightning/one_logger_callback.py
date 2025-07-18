"""
OneLogger callback for NeMo training.

This module provides a callback that integrates OneLogger telemetry with NeMo training.
"""

import functools
import time
from typing import Any, Dict, List, Optional, Type

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
from lightning.pytorch.utilities import rank_zero_only
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


# Wrapper functions for OneLogger callbacks
def get_onelogger_callbacks(name: str):
    """Get the OneLogger callbacks module if available."""
    if not HAVE_ONELOGGER:

        def _noop(*args, **kwargs):
            pass

        return _noop
    if hasattr(CB, name):
        return getattr(CB, name)
    else:
        raise AttributeError(f"OneLogger has no attribute {name}")


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
        """Track a timing event with automatic start/end timing.

        This method automatically handles the start and end timing for an event.
        It detects start/end events based on event names ending with "_start" or "_end".

        Args:
            event_type: Type of event (e.g., 'model_init_start', 'model_init_end', 'dataloader_init')
        """
        if current_time_ms is None:
            current_time_ms = get_current_time_msec()

        event = {'name': event_type, 'time_ms': current_time_ms}

        if not self._one_logger_available:
            self._pending_events.append(event)
        else:
            self._log_event(event)

    def set_one_logger_available(self, available: bool = True):
        """Set whether OneLogger is available and process pending events.

        Args:
            available: Whether OneLogger is now available
        """
        self._one_logger_available = available
        if available and self._pending_events:
            # Process all pending events
            for event in self._pending_events:
                self._log_event(event)
            self._pending_events.clear()

    @classmethod
    def mark_one_logger_available(cls):
        """Class method to mark OneLogger as available globally."""
        instance = cls.get_instance()
        instance.set_one_logger_available(True)

    def _log_event(self, event: Dict[str, Any]):
        """Log an event using OneLogger callbacks.

        Args:
            event: Event data containing name, time_ms
        """
        # If nv-one-logger is not available, or OneLogger is not yet initialized, skip logging
        if not HAVE_ONELOGGER or not self._one_logger_available:
            return

        # Handle start/end event pairs
        event_name = event['name']
        time_ms = event['time_ms']

        if event_name.endswith('_start'):
            get_onelogger_callbacks(event_name)(start_time_msec=time_ms)
        elif event_name.endswith('_end'):
            get_onelogger_callbacks(event_name)(finish_time_msec=time_ms)
        else:
            raise ValueError(f"Invalid event name for api: {event_name}")


class OneLoggerNeMoCallback(Callback):
    """
    NeMo callback that integrates with OneLogger v2 for tracking metrics.

    This callback implements NeMo's callback group API and internally
    uses OneLogger's training telemetry functionality to track metrics.
    """

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
        return get_onelogger_callbacks(name)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when training begins."""
        # Extract necessary information from the trainer
        current_step = trainer.global_step
        max_steps = trainer.max_steps if hasattr(trainer, 'max_steps') else 0

        get_onelogger_callbacks("on_train_start")(
            train_iterations_start=current_step, train_iterations_target_or_fn=max_steps
        )

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        get_onelogger_callbacks("on_training_single_iteration_start")()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        get_onelogger_callbacks("on_training_single_iteration_end")()

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        get_onelogger_callbacks("on_validation_start")()

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._validation_batch_exists:
            get_onelogger_callbacks("on_validation_single_iteration_end")()
            self._validation_batch_exists = False
        get_onelogger_callbacks("on_validation_end")()

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self._validation_batch_exists:
            get_onelogger_callbacks("on_validation_single_iteration_end")()
        self._validation_batch_exists = True
        get_onelogger_callbacks("on_validation_single_iteration_start")()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        get_onelogger_callbacks("on_validation_single_iteration_end")()

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when training ends."""
        if self._train_active:
            get_onelogger_callbacks("on_train_end")()
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

    tracker = OneLoggerTimingTracker.get_instance()

    @functools.wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        # Check if this instance has already been initialized to prevent duplicate callbacks
        # in inheritance chains
        if hasattr(self, '_one_logger_init_started'):
            # This instance is already being initialized, skip the callbacks
            return original_init(self, *args, **kwargs)
        
        # Mark this instance as being initialized
        self._one_logger_init_started = True
        
        print("NeMo CB: wrapped_init for class", cls.__name__)
        tracker.track_event(start_callback)
        result = original_init(self, *args, **kwargs)
        tracker.track_event(end_callback)
        return result

    # Mark as wrapped to prevent double wrapping
    wrapped_init._one_logger_wrapped = True
    cls.__init__ = wrapped_init


def init_one_logger(v1_config: Dict[str, Any], trainer: Trainer = None, enable_onelogger: bool = True):
    """Initialize OneLogger with v1 config and optionally add callback to trainer.

    Args:
        v1_config: V1-style configuration dictionary
        trainer: Optional PyTorch Lightning trainer to add callback to
        enable_onelogger: Whether to enable OneLogger (default: True)
    """
    if not HAVE_ONELOGGER or not enable_onelogger:
        return

    # Convert v1 config to v2 config using the adapter
    training_telemetry_config, wandb_config = ConfigAdapter.convert_to_v2_config(v1_config)

    # Configure OneLogger using v1 adapter with async wandb exporter
    exporter = V1CompatibleWandbExporterAsync(
        training_telemetry_config=training_telemetry_config,
        wandb_config=wandb_config,
    )
    TrainingTelemetryProvider.instance().with_base_telemetry_config(training_telemetry_config).with_exporter(
        exporter
    ).configure_provider()

    OneLoggerTimingTracker.mark_one_logger_available()

    # Add the OneLogger callback to the trainer if provided
    if trainer is not None:
        # Check if OneLoggerNeMoCallback is already in the trainer's callbacks
        has_onelogger_callback = any(isinstance(callback, OneLoggerNeMoCallback) for callback in trainer.callbacks)

        if not has_onelogger_callback:
            # Create the callback with metadata
            onelogger_callback = OneLoggerNeMoCallback()
            trainer.callbacks.append(onelogger_callback)
