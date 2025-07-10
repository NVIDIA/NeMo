"""
OneLogger callback for NeMo training.

This module provides a callback that integrates OneLogger telemetry with NeMo training.
"""

import functools
import logging
import time
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
    
    def track_event(self, event_type: str):
        """Track a timing event with automatic start/end timing.
        
        This method automatically handles the start and end timing for an event.
        It detects start/end events based on event names ending with "_start" or "_end".
        
        Args:
            event_type: Type of event (e.g., 'model_init_start', 'model_init_end', 'dataloader_init')
        """
        current_time_ms = get_current_time_msec()
        
        # Check if this is a start or end event
        if event_type.endswith('_start'):
            # This is a start event - store it for later pairing with end event
            event = {
                'type': event_type,
                'start_time_ms': current_time_ms,
                'end_time_ms': None,
                'is_start': True
            }
        elif event_type.endswith('_end'):
            # This is an end event - find and complete the corresponding start event
            base_event_type = event_type[:-4]  # Remove '_end' suffix
            start_event_type = base_event_type + '_start'
            
            # Find the corresponding start event
            for pending_event in self._pending_events:
                if (pending_event['type'] == start_event_type and 
                    pending_event['end_time_ms'] is None and
                    pending_event.get('is_start', False)):
                    # Complete the start event with end time
                    pending_event['end_time_ms'] = current_time_ms
                    pending_event['is_start'] = False
                    
                    # Log the completed event
                    if self._one_logger_available:
                        self._log_event(pending_event)
                    return
            
            # If no matching start event found, log as immediate event
            event = {
                'type': event_type,
                'start_time_ms': current_time_ms,
                'end_time_ms': current_time_ms,
                'is_start': False
            }
        else:
            # This is an immediate event (neither start nor end)
            event = {
                'type': event_type,
                'start_time_ms': current_time_ms,
                'end_time_ms': current_time_ms,
                'is_start': False
            }
        
        if self._one_logger_available:
            self._log_event(event)
        else:
            self._pending_events.append(event)
    

    
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
            event: Event data containing type, start_time_ms, and end_time_ms
        """
        try:
            # Import OneLogger callbacks
            from nv_one_logger.training_telemetry.api import callbacks as one_logger_callbacks
            from nv_one_logger.training_telemetry.api import training_telemetry_provider
            
            # Check if OneLogger is actually ready
            if not training_telemetry_provider.TrainingTelemetryProvider.instance().one_logger_ready:
                return
            
            event_type = event['type']
            start_time_ms = event['start_time_ms']
            end_time_ms = event.get('end_time_ms')
            
            # Handle start/end event pairs
            if event_type.endswith('_start'):
                # This is a start event - log the start
                base_event_type = event_type[:-6]  # Remove '_start' suffix
                if base_event_type == 'model_init':
                    one_logger_callbacks.on_model_init_start(start_time_msec=start_time_ms)
                elif base_event_type == 'dataloader_init':
                    one_logger_callbacks.on_dataloader_init_start(start_time_msec=start_time_ms)
                elif base_event_type == 'optimizer_init':
                    one_logger_callbacks.on_optimizer_init_start(start_time_msec=start_time_ms)
                elif base_event_type == 'load_checkpoint':
                    one_logger_callbacks.on_load_checkpoint_start(start_time_msec=start_time_ms)
                elif base_event_type == 'save_checkpoint':
                    one_logger_callbacks.on_save_checkpoint_start(start_time_msec=start_time_ms)
                elif base_event_type == 'app':
                    one_logger_callbacks.on_app_start(start_time_msec=start_time_ms)
                    
            elif event_type.endswith('_end'):
                # This is an end event - log the end
                base_event_type = event_type[:-4]  # Remove '_end' suffix
                if base_event_type == 'model_init':
                    one_logger_callbacks.on_model_init_end(finish_time_msec=end_time_ms)
                elif base_event_type == 'dataloader_init':
                    one_logger_callbacks.on_dataloader_init_end(finish_time_msec=end_time_ms)
                elif base_event_type == 'optimizer_init':
                    one_logger_callbacks.on_optimizer_init_end(finish_time_msec=end_time_ms)
                elif base_event_type == 'load_checkpoint':
                    one_logger_callbacks.on_load_checkpoint_end(finish_time_msec=end_time_ms)
                elif base_event_type == 'save_checkpoint':
                    one_logger_callbacks.on_save_checkpoint_end(finish_time_msec=end_time_ms)
                elif base_event_type == 'app':
                    one_logger_callbacks.on_app_end(finish_time_msec=end_time_ms)
                    
            else:
                raise 
                elif event_type == 'app_end':
                    # Log only end
                    one_logger_callbacks.on_app_end(finish_time_msec=start_time_ms)
                    
        except ImportError:
            # OneLogger not available, ignore
            pass
        except Exception as e:
            # Log error but don't fail
            logging.debug(f"Failed to log OneLogger event {event['type']}: {e}")


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
        print(f"OneLogger NeMo CB: __init__ called with log_interval={log_interval}")
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
            print(f"  ✓ Config loaded: app_name={self.app_name}, perf_tag={self.perf_tag}, global_batch_size={self.global_batch_size}")
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
        print(f"OneLogger NeMo CB: __getattr__ called for '{name}'")
        # Check if the method exists in the OneLogger callbacks module
        if hasattr(CB, name):
            # Get the original method
            original_method = getattr(CB, name)

            # Create a wrapper that adds rank_zero_only decorator
            @functools.wraps(original_method)
            def wrapper(*args, **kwargs):
                print(f"OneLogger NeMo CB: Forwarding call to CB.{name}")
                return rank_zero_only(original_method)(*args, **kwargs)

            return wrapper

        # If not found, raise AttributeError as normal
        print(f"OneLogger NeMo CB: Attribute '{name}' not found in CB module")
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when training begins."""
        print(f"OneLogger NeMo CB: on_train_start called")
        # Extract necessary information from the trainer
        current_step = trainer.global_step
        max_steps = trainer.max_steps if hasattr(trainer, 'max_steps') else 0
        print(f"  ✓ Trainer info: current_step={current_step}, max_steps={max_steps}")

        CB.on_train_start(train_iterations_start=current_step, train_iterations_target_or_fn=max_steps)
        print(f"  ✓ Called CB.on_train_start")

    @rank_zero_only
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print(f"OneLogger NeMo CB: on_train_end called")
        CB.on_train_end()
        print(f"  ✓ Called CB.on_train_end")

    @rank_zero_only
    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        print(f"OneLogger NeMo CB: on_train_batch_start called with batch_idx={batch_idx}")
        CB.on_training_single_iteration_start()
        print(f"  ✓ Called CB.on_training_single_iteration_start")

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        print(f"OneLogger NeMo CB: on_train_batch_end called with batch_idx={batch_idx}")
        CB.on_training_single_iteration_end()
        print(f"  ✓ Called CB.on_training_single_iteration_end")

    @rank_zero_only
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print(f"OneLogger NeMo CB: on_validation_start called")
        CB.on_validation_start()
        print(f"  ✓ Called CB.on_validation_start")

    @rank_zero_only
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print(f"OneLogger NeMo CB: on_validation_end called")
        CB.on_validation_end()
        print(f"  ✓ Called CB.on_validation_end")

    @rank_zero_only
    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        print(f"OneLogger NeMo CB: on_validation_batch_start called with batch_idx={batch_idx}, dataloader_idx={dataloader_idx}")
        CB.on_validation_single_iteration_start()
        print(f"  ✓ Called CB.on_validation_single_iteration_start")

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
        print(f"OneLogger NeMo CB: on_validation_batch_end called with batch_idx={batch_idx}, dataloader_idx={dataloader_idx}")
        CB.on_validation_single_iteration_end()
        print(f"  ✓ Called CB.on_validation_single_iteration_end")

    @rank_zero_only
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Called when fit, test, or predict begins."""
        print(f"OneLogger NeMo CB: setup called with stage={stage}")
        super().setup(trainer, pl_module, stage)
        print(f"  ✓ Called super().setup")

    @rank_zero_only
    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Called when fit, test, or predict ends."""
        print(f"OneLogger NeMo CB: teardown called with stage={stage}")
        super().teardown(trainer, pl_module, stage)
        print(f"  ✓ Called super().teardown")

    @rank_zero_only
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when fit begins."""
        print(f"OneLogger NeMo CB: on_fit_start called")
        super().on_fit_start(trainer, pl_module)
        print(f"  ✓ Called super().on_fit_start")

    @rank_zero_only
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when fit ends."""
        print(f"OneLogger NeMo CB: on_fit_end called")
        super().on_fit_end(trainer, pl_module)
        print(f"  ✓ Called super().on_fit_end")

    @rank_zero_only
    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> None:
        """Called when a checkpoint is saved."""
        print(f"OneLogger NeMo CB: on_save_checkpoint called")
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        print(f"  ✓ Called super().on_save_checkpoint")

    @rank_zero_only
    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> None:
        """Called when a checkpoint is loaded."""
        print(f"OneLogger NeMo CB: on_load_checkpoint called")
        super().on_load_checkpoint(trainer, pl_module, checkpoint)
        print(f"  ✓ Called super().on_load_checkpoint")

    def on_app_start(self, start_time_msec: float = None) -> None:
        """Called at the start of the application."""
        print(f"OneLogger NeMo CB: on_app_start called with start_time_msec={start_time_msec}")
        if start_time_msec is None:
            start_time_msec = get_current_time_msec()
        CB.on_app_start(start_time_msec=start_time_msec)
        print(f"  ✓ Called CB.on_app_start")

    def on_app_end(self) -> None:
        """Called at the end of the application."""
        print(f"OneLogger NeMo CB: on_app_end called")
        CB.on_app_end()
        print(f"  ✓ Called CB.on_app_end")

    def on_dataloader_init_start(self) -> None:
        """Called at the start of data loader initialization."""
        print(f"OneLogger NeMo CB: on_dataloader_init_start called")
        # OneLogger doesn't have a specific callback for this, but we can log it
        print(f"  ✓ Data loader initialization started")

    def on_dataloader_init_end(self) -> None:
        """Called at the end of data loader initialization."""
        print(f"OneLogger NeMo CB: on_dataloader_init_end called")
        # OneLogger doesn't have a specific callback for this, but we can log it
        print(f"  ✓ Data loader initialization completed")

    def on_model_init_start(self) -> None:
        """Called at the start of model initialization."""
        print(f"OneLogger NeMo CB: on_model_init_start called")
        # OneLogger doesn't have a specific callback for this, but we can log it
        print(f"  ✓ Model initialization started")

    def on_model_init_end(self) -> None:
        """Called at the end of model initialization."""
        print(f"OneLogger NeMo CB: on_model_init_end called")
        # OneLogger doesn't have a specific callback for this, but we can log it
        print(f"  ✓ Model initialization completed")

    def on_optimizer_init_start(self) -> None:
        """Called at the start of optimizer initialization."""
        print(f"OneLogger NeMo CB: on_optimizer_init_start called")
        # OneLogger doesn't have a specific callback for this, but we can log it
        print(f"  ✓ Optimizer initialization started")

    def on_optimizer_init_end(self) -> None:
        """Called at the end of optimizer initialization."""
        print(f"OneLogger NeMo CB: on_optimizer_init_end called")
        # OneLogger doesn't have a specific callback for this, but we can log it
        print(f"  ✓ Optimizer initialization completed")

    def on_load_checkpoint_start(self) -> None:
        """Called at the start of checkpoint loading."""
        print(f"OneLogger NeMo CB: on_load_checkpoint_start called")
        # OneLogger doesn't have a specific callback for this, but we can log it
        print(f"  ✓ Checkpoint loading started")

    def on_load_checkpoint_end(self) -> None:
        """Called at the end of checkpoint loading."""
        print(f"OneLogger NeMo CB: on_load_checkpoint_end called")
        # OneLogger doesn't have a specific callback for this, but we can log it
        print(f"  ✓ Checkpoint loading completed")

    def on_save_checkpoint_start(self, iteration: int = 0) -> None:
        """Called at the start of checkpoint saving."""
        print(f"OneLogger NeMo CB: on_save_checkpoint_start called with iteration={iteration}")
        try:
            CB.on_save_checkpoint_start(global_step=iteration)
            print(f"  ✓ Called CB.on_save_checkpoint_start")
        except ImportError:
            # OneLogger not available, ignore
            pass
        except Exception as e:
            # Log error but don't fail
            print(f"  ✗ Failed to call OneLogger on_save_checkpoint_start: {e}")

    def on_save_checkpoint_end(self, iteration: int = 0) -> None:
        """Called at the end of checkpoint saving."""
        print(f"OneLogger NeMo CB: on_save_checkpoint_end called with iteration={iteration}")
        try:
            CB.on_save_checkpoint_end()
            print(f"  ✓ Called CB.on_save_checkpoint_end")
        except ImportError:
            # OneLogger not available, ignore
            pass
        except Exception as e:
            # Log error but don't fail
            print(f"  ✗ Failed to call OneLogger on_save_checkpoint_end: {e}")

    def on_save_checkpoint_success(self, iteration: int = 0) -> None:
        """Called when checkpoint saving is successful."""
        print(f"OneLogger NeMo CB: on_save_checkpoint_success called with iteration={iteration}")
        try:
            CB.on_save_checkpoint_success(global_step=iteration)
            print(f"  ✓ Called CB.on_save_checkpoint_success")
        except ImportError:
            # OneLogger not available, ignore
            pass
        except Exception as e:
            # Log error but don't fail
            print(f"  ✗ Failed to call OneLogger on_save_checkpoint_success: {e}")
