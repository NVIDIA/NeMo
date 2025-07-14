"""
OneLogger callback for NeMo training.

This module provides a callback that integrates OneLogger telemetry with NeMo training.
"""

import functools
import logging
import time
from typing import Any, Dict, List, Optional, Type

import nv_one_logger.training_telemetry.api.callbacks as CB  # TODO: make the import safe -- the lib may not be installed, and it's ok to skip for users.
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core import LightningModule
from pytorch_lightning.plugins.io import AsyncCheckpointIO
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT


def get_current_time_msec() -> float:
    """Get current time in milliseconds since epoch.

    Returns:
        float: Current time in milliseconds since epoch
    """
    return time.time() * 1000


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
        if not self._one_logger_available:
            return

        # Handle start/end event pairs
        event_name = event['name']
        time_ms = event['time_ms']

        if not hasattr(CB, event_name):
            raise ValueError(f"Invalid event name for api: {event_name}")

        if event_name.endswith('_start'):
            getattr(CB, event_name)(start_time_msec=time_ms)
        elif event_name.endswith('_end'):
            getattr(CB, event_name)(finish_time_msec=time_ms)
        else:
            raise ValueError(f"Invalid event name for api: {event_name}")


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
            print(
                f"  ✓ Config loaded: app_name={self.app_name}, perf_tag={self.perf_tag}, global_batch_size={self.global_batch_size}"
            )
        else:
            self.app_name = ""
            self.perf_tag = ""
            self.session_tag = ""
            self.global_batch_size = 0
            print(f"  ⚠ No callback_config provided, using defaults")

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

            return original_method

        # If not found, raise AttributeError as normal
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )  # TODO: remove this to not raise exception

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when training begins."""
        # Extract necessary information from the trainer
        current_step = trainer.global_step
        max_steps = trainer.max_steps if hasattr(trainer, 'max_steps') else 0

        CB.on_train_start(train_iterations_start=current_step, train_iterations_target_or_fn=max_steps)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        CB.on_training_single_iteration_start()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        CB.on_training_single_iteration_end()

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        CB.on_validation_start()

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        CB.on_validation_end()

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        CB.on_validation_single_iteration_start()

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
        tracker.track_event(start_callback)
        result = original_init(self, *args, **kwargs)
        tracker.track_event(end_callback)
        return result

    # Mark as wrapped to prevent double wrapping
    wrapped_init._one_logger_wrapped = True
    cls.__init__ = wrapped_init
