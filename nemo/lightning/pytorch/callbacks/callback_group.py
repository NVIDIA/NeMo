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
from typing import Callable

from lightning.pytorch.callbacks import Callback


class CallbackGroup:
    """A class for hosting a collection of callback objects.

    It is used to execute callback functions of multiple callback objects with the same method name.
    When callbackgroup.func(args) is executed, internally it loops through the objects in
    self._callbacks and runs self._callbacks[0].func(args), self._callbacks[1].func(args), etc.
    The method name and arguments should match.

    Attributes:
        _callbacks (list[Callback]): List of callback objects.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> 'CallbackGroup':
        """Get the singleton instance of the CallbackGroup.
        Args:
            cls (CallbackGroup): The class of the CallbackGroup.
        Returns:
            CallbackGroup: The singleton instance of the CallbackGroup.
        """
        if cls._instance is None:
            cls._instance = CallbackGroup()
        return cls._instance

    def __init__(self) -> None:
        """Initializes the list of callback objects."""
        self._callbacks = []

    def register(self, callback: Callback) -> None:
        """Register a callback to the callback group.

        Args:
            callback (Callback): The callback to register.
        """
        self._callbacks.append(callback)

    def __getattr__(self, method_name: str) -> Callable:
        """Loops through the callback objects to call the corresponding callback function.

        Args:
            method_name (str): Callback method name.
        """

        def multi_callback_wrapper(*args, **kwargs) -> None:
            for callback in self._callbacks:
                assert hasattr(callback, method_name)
                method = getattr(callback, method_name)
                assert callable(method)
                _ = method(*args, **kwargs)

        return multi_callback_wrapper

    @property
    def callbacks(self):
        """Return callbacks in order.

        Returns:
            list: callback objects
        """
        return self._callbacks


class Callback(Callback):
    """The base class for all callbacks. It inherits the pytorch lightning callback so the callback can be also passed to PTL trainer to reuse.
    Below list extra callback functions in NeMo.
    """

    def on_dataloader_init_start(self):
        """Called at the start of the data loading."""

    def on_dataloader_init_end(self):
        """Called at the end of the data loading."""

    def on_model_init_start(self):
        """Called at the start of the model initialization."""

    def on_model_init_end(self):
        """Called at the end of the model initialization."""

    def on_optimizer_init_start(self) -> None:
        """Called at the beginning of optimizer initialization."""

    def on_optimizer_init_end(self) -> None:
        """Called at the end of optimizer initialization."""

    def on_load_checkpoint_start(self) -> None:
        """Called at the beginning of loading checkpoint."""

    def on_load_checkpoint_end(self) -> None:
        """Called at the end of loading checkpoint."""

    def on_save_checkpoint_start(self, iteration: int = 0) -> None:
        """Called when start saving a checkpoint."""

    def on_save_checkpoint_end(self, iteration: int = 0) -> None:
        """Called when saving checkpoint (sync part) call ends."""

    def on_save_checkpoint_success(self, iteration: int = 0) -> None:
        """Called when checkpoint is saved successfully."""


CB_WRAP_RULES = {
    # The function name is the name of the method to wrap.
    # The start_hook and end_hook are the names of the methods to call before and after the original method.
    # The callback_method_name is the name of the method to call in the callback group.
    # Example:
    # function name: {
    #     "start_hook": callback_method_name,
    #     "end_hook": callback_method_name
    # }
    "setup_training_data": {"start_hook": "on_dataloader_init_start", "end_hook": "on_dataloader_init_end"},
    "setup_optimization": {"start_hook": "on_optimizer_init_start", "end_hook": "on_optimizer_init_end"},
    "restore_from_pretrained_models": {"start_hook": "on_load_checkpoint_start", "end_hook": "on_load_checkpoint_end"},
    "__init__": {"start_hook": "on_model_init_start", "end_hook": "on_model_init_end"},
    "configure_optimizers": {"start_hook": "on_optimizer_init_start", "end_hook": "on_optimizer_init_end"},
    "setup_training_dataloader": {"start_hook": "on_dataloader_init_start", "end_hook": "on_dataloader_init_end"},
}


def _make_callback_wrapped_method(original_method):
    """Wrap a method with the start and end hooks of the callback group.

    Args:
        original_method (Callable): The original method to wrap.
        hooks (dict): The hooks to call.
    """
    callback_group = CallbackGroup.get_instance()
    hooks = CB_WRAP_RULES.get(original_method.__name__)

    is_classmethod = isinstance(original_method, classmethod)

    if not hooks:
        return original_method

    @functools.wraps(original_method)
    def wrapped_instance_method(self, *args, **kwargs):
        if hasattr(callback_group, hooks["start_hook"]):
            getattr(callback_group, hooks["start_hook"])()
        result = original_method(self, *args, **kwargs)
        if hasattr(callback_group, hooks["end_hook"]):
            getattr(callback_group, hooks["end_hook"])()
        return result

    @functools.wraps(original_method)
    def wrapped_class_method(*args, **kwargs):
        if hasattr(callback_group, hooks["start_hook"]):
            getattr(callback_group, hooks["start_hook"])()
        result = original_method(*args, **kwargs)
        if hasattr(callback_group, hooks["end_hook"]):
            getattr(callback_group, hooks["end_hook"])()
        return result

    if is_classmethod:
        return classmethod(wrapped_class_method)
    else:
        return wrapped_instance_method


def wrap_methods_with_callbacks(cls) -> None:
    """Wrap class/instance methods with the start and end hooks of the callback group.

    Args:
        cls (type): The class to wrap the methods of.
    """
    for method_name in CB_WRAP_RULES.keys():
        if method_name in cls.__dict__:
            original_method = cls.__dict__[method_name]
            cls.__dict__[method_name] = _make_callback_wrapped_method(original_method)
