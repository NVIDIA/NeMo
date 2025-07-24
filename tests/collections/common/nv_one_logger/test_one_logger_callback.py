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

import time
from unittest.mock import MagicMock, patch

import pytest

from nemo.lightning.one_logger_callback import (
    OneLoggerNeMoCallback,
    get_current_time_msec,
    get_one_logger_callbacks,
    hook_class_init_with_callbacks,
)


class TestOneLoggerCallback:
    """Test cases for OneLogger callback functionality."""

    @pytest.mark.unit
    def test_get_current_time_msec(self):
        """Test that get_current_time_msec returns time in milliseconds."""
        time_before = time.time() * 1000
        result = get_current_time_msec()
        time_after = time.time() * 1000

        assert isinstance(result, float)
        assert time_before <= result <= time_after

    @pytest.mark.unit
    def test_get_onelogger_callbacks_no_onelogger(self):
        """Test get_one_logger_callbacks when OneLogger is not available."""
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', False):
            # Should return None when OneLogger is not available
            result = get_one_logger_callbacks("test_callback")
            assert result is None

    @pytest.mark.unit
    def test_one_logger_nemo_callback_initialization(self):
        """Test OneLoggerNeMoCallback initialization."""
        callback = OneLoggerNeMoCallback()
        assert isinstance(callback, OneLoggerNeMoCallback)

    @pytest.mark.unit
    def test_one_logger_nemo_callback_getattr(self):
        """Test __getattr__ method of OneLoggerNeMoCallback."""
        callback = OneLoggerNeMoCallback()

        with (
            patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True),
            patch('nemo.lightning.one_logger_callback.CB') as mock_cb,
        ):
            mock_callback = MagicMock()
            mock_cb.test_method = mock_callback

            result = callback.test_method
            assert result == mock_callback

    @pytest.mark.unit
    def test_training_cycle_all_callbacks(self):
        """Test a complete training cycle with all callbacks called in order."""
        callback = OneLoggerNeMoCallback()
        trainer = MagicMock()
        trainer.global_step = 10
        trainer.max_steps = 1000
        pl_module = MagicMock()
        batch = MagicMock()
        outputs = MagicMock()

        # Track all callback calls in order
        callback_calls = []

        with (
            patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True),
            patch('nemo.lightning.one_logger_callback.CB') as mock_cb,
        ):
            # Mock the CB module to return our mock callback
            mock_cb.test_start = MagicMock()
            mock_cb.test_end = MagicMock()
            mock_cb.on_train_start = MagicMock()
            mock_cb.on_training_single_iteration_start = MagicMock()
            mock_cb.on_training_single_iteration_end = MagicMock()
            mock_cb.on_validation_start = MagicMock()
            mock_cb.on_validation_single_iteration_start = MagicMock()
            mock_cb.on_validation_single_iteration_end = MagicMock()
            mock_cb.on_validation_end = MagicMock()
            mock_cb.on_train_end = MagicMock()

            # Set up the side effect to track calls
            def side_effect(callback_name, *args, **kwargs):
                callback_calls.append(callback_name)
                return MagicMock()

            with patch('nemo.lightning.one_logger_callback.get_one_logger_callbacks', side_effect=side_effect):
                # Simulate training cycle
                callback.on_train_start(trainer, pl_module)
                callback.on_train_batch_start(trainer, pl_module, batch, 0)
                callback.on_train_batch_end(trainer, pl_module, outputs, batch, 0)
                callback.on_validation_start(trainer, pl_module)
                callback.on_validation_batch_start(trainer, pl_module, batch, 0, 0)
                callback.on_validation_batch_end(trainer, pl_module, outputs, batch, 0, 0)
                callback.on_validation_end(trainer, pl_module)
                callback.on_train_end(trainer, pl_module)

                # Verify all callbacks were called in the expected order
                expected_calls = [
                    "on_train_start",
                    "on_training_single_iteration_start",
                    "on_training_single_iteration_end",
                    "on_validation_start",
                    "on_validation_single_iteration_start",
                    "on_validation_single_iteration_end",
                    "on_validation_end",
                    "on_train_end",
                ]

                assert callback_calls == expected_calls

    @pytest.mark.unit
    def test_hook_class_init_with_callbacks_no_init(self):
        """Test hook_class_init_with_callbacks with class that has no __init__."""

        class TestClass:
            pass

        hook_class_init_with_callbacks(TestClass, "start_callback", "end_callback")
        # Should not raise any exception

    @pytest.mark.unit
    def test_hook_class_init_with_callbacks(self):
        """Test hook_class_init_with_callbacks functionality."""

        class TestClass:
            def __init__(self, value=0):
                self.value = value

        original_init = TestClass.__init__

        hook_class_init_with_callbacks(TestClass, "test_start", "test_end")

        # Check that __init__ was wrapped
        assert TestClass.__init__ != original_init
        assert hasattr(TestClass.__init__, '_one_logger_wrapped')
        assert TestClass.__init__._one_logger_wrapped is True

    @pytest.mark.unit
    def test_hook_class_init_with_callbacks_double_wrap(self):
        """Test that hook_class_init_with_callbacks doesn't double wrap."""

        class TestClass:
            def __init__(self, value=0):
                self.value = value

        hook_class_init_with_callbacks(TestClass, "test_start", "test_end")
        wrapped_init = TestClass.__init__

        # Try to wrap again
        hook_class_init_with_callbacks(TestClass, "test_start2", "test_end2")

        # Should still be the same wrapped function
        assert TestClass.__init__ == wrapped_init
