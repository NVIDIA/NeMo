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

    def __init__(self, callbacks: list[Callback] | None) -> None:
        """Initializes the list of callback objects.

        Args:
            callbacks (list[Callback]): List of callbacks.
        """
        self._callbacks = callbacks or []

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
