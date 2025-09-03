# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import functools
from typing import Callable, List, Optional

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback as PTLCallback


class CallbackGroup:
    """A singleton registry to host and fan-out lifecycle callbacks.

    Other code should call methods on this group (e.g., `on_model_init_start`).
    The group will iterate all registered callbacks and, if a callback implements
    the method, invoke it with the provided arguments.
    """

    _instance: Optional['CallbackGroup'] = None

    @classmethod
    def get_instance(cls) -> 'CallbackGroup':
        if cls._instance is None:
            cls._instance = CallbackGroup()
        return cls._instance

    def __init__(self) -> None:
        self._callbacks: List[BaseCallback] = []

    def register(self, callback: 'BaseCallback') -> None:
        """Register a callback instance in FIFO order."""
        self._callbacks.append(callback)

    def update_config(self, nemo_version: str, trainer: Trainer, **kwargs) -> None:
        """Update configuration across all registered callbacks and attach them to trainer.

        Args:
            nemo_version: Version key (e.g., 'v1' or 'v2') for downstream config builders.
            trainer: Lightning Trainer to which callbacks should be attached if missing.
            **kwargs: Forwarded to each callback's update_config implementation.
        """
        # Forward update to each callback that supports update_config
        for cb in self._callbacks:
            if hasattr(cb, 'update_config'):
                method = getattr(cb, 'update_config')
                if callable(method):
                    method(nemo_version=nemo_version, trainer=trainer, **kwargs)
            trainer.callbacks.append(cb)

    @property
    def callbacks(self) -> List['BaseCallback']:
        return self._callbacks

    def __getattr__(self, method_name: str) -> Callable:
        """Dynamically create a dispatcher for unknown attributes.

        Any attribute access is treated as a lifecycle method name.
        When invoked, the dispatcher will call that method on each registered
        callback if it exists.
        """

        def dispatcher(*args, **kwargs):
            for cb in self._callbacks:
                if hasattr(cb, method_name):
                    method = getattr(cb, method_name)
                    if callable(method):
                        method(*args, **kwargs)

        return dispatcher


class BaseCallback(PTLCallback):
    """Base callback ABC for NeMo lifecycle hooks (extends PTL callback).

    Implementers may override any subset of the following methods. All are
    optional no-op defaults to keep implementations lightweight.
    """

    # App lifecycle
    def on_app_start(self, *args, **kwargs) -> None:
        pass

    def on_app_end(self, *args, **kwargs) -> None:
        pass

    # Model lifecycle
    def on_model_init_start(self, *args, **kwargs) -> None:
        pass

    def on_model_init_end(self, *args, **kwargs) -> None:
        pass

    # Dataloader lifecycle
    def on_dataloader_init_start(self, *args, **kwargs) -> None:
        pass

    def on_dataloader_init_end(self, *args, **kwargs) -> None:
        pass

    # Optimizer lifecycle
    def on_optimizer_init_start(self, *args, **kwargs) -> None:
        pass

    def on_optimizer_init_end(self, *args, **kwargs) -> None:
        pass

    # Checkpoint lifecycle
    def on_load_checkpoint_start(self, *args, **kwargs) -> None:
        pass

    def on_load_checkpoint_end(self, *args, **kwargs) -> None:
        pass

    def on_save_checkpoint_start(self, *args, **kwargs) -> None:
        pass

    def on_save_checkpoint_end(self, *args, **kwargs) -> None:
        pass

    def on_save_checkpoint_success(self, *args, **kwargs) -> None:
        pass

    # Configuration update
    def update_config(self, *args, **kwargs) -> None:
        """Update callback-specific configuration after initialization.

        Example: telemetry/training config that depends on `Trainer` or runtime state.
        """
        pass


def hook_class_init_with_callbacks(cls, start_callback: str, end_callback: str) -> None:
    """Hook a class's __init__ to emit CallbackGroup start/end hooks.

    Args:
        cls (type): Class whose __init__ should be wrapped.
        start_callback (str): CallbackGroup method to call before __init__.
        end_callback (str): CallbackGroup method to call after __init__.
    """
    if not hasattr(cls, '__init__'):
        return

    original_init = cls.__init__

    if getattr(original_init, '_callback_group_wrapped', False):
        return

    @functools.wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        group = CallbackGroup.get_instance()
        if hasattr(group, start_callback):
            getattr(group, start_callback)()
        result = original_init(self, *args, **kwargs)
        if hasattr(group, end_callback):
            getattr(group, end_callback)()
        return result

    wrapped_init._callback_group_wrapped = True
    cls.__init__ = wrapped_init


# Eagerly create the singleton on import so that early callers can use it
CallbackGroup.get_instance()

__all__ = ['CallbackGroup', 'BaseCallback', 'hook_class_init_with_callbacks']
