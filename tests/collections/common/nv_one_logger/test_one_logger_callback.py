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
    _get_onelogger_callbacks_function,
    get_current_time_msec,
    get_one_logger_callbacks,
    hook_class_init_with_callbacks,
    init_one_logger,
    update_one_logger_config,
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
    def test_get_onelogger_callbacks_with_onelogger(self):
        """Test get_one_logger_callbacks when OneLogger is available."""
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            with patch('nemo.lightning.one_logger_callback.CB') as mock_cb:
                mock_callback = MagicMock()
                mock_cb.test_callback = mock_callback

                result = get_one_logger_callbacks("test_callback", arg1="value1")

                mock_callback.assert_called_once_with(arg1="value1")
                assert result == mock_callback.return_value

    @pytest.mark.unit
    def test_get_onelogger_callbacks_attribute_error(self):
        """Test get_one_logger_callbacks when callback doesn't exist."""
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            with patch('nemo.lightning.one_logger_callback.CB') as mock_cb:
                # Remove the attribute to simulate it not existing
                delattr(mock_cb, 'nonexistent_callback')

                with pytest.raises(AttributeError, match="OneLogger has no attribute nonexistent_callback"):
                    get_one_logger_callbacks("nonexistent_callback")

    @pytest.mark.unit
    def test_get_onelogger_callbacks_function_no_onelogger(self):
        """Test _get_onelogger_callbacks_function when OneLogger is not available."""
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', False):
            function = _get_onelogger_callbacks_function("test_callback")
            function("arg1", kwarg1="value1")  # Call function but don't use return value
            # The function should not raise any exception

    @pytest.mark.unit
    def test_get_onelogger_callbacks_function_with_onelogger(self):
        """Test _get_onelogger_callbacks_function when OneLogger is available."""
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            with patch('nemo.lightning.one_logger_callback.CB') as mock_cb:
                mock_callback = MagicMock()
                mock_cb.test_callback = mock_callback

                function = _get_onelogger_callbacks_function("test_callback")
                function("arg1", kwarg1="value1")  # Call function but don't use return value

                assert function == mock_callback
                # Verify the mock was called with the expected arguments
                mock_callback.assert_called_once_with("arg1", kwarg1="value1")

    @pytest.mark.unit
    def test_one_logger_nemo_callback_initialization(self):
        """Test OneLoggerNeMoCallback initialization."""
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            with patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider:
                mock_instance = MagicMock()
                mock_provider.instance.return_value = mock_instance
                callback = OneLoggerNeMoCallback(mock_instance)
                assert isinstance(callback, OneLoggerNeMoCallback)

    @pytest.mark.unit
    def test_one_logger_nemo_callback_getattr(self):
        """Test __getattr__ method of OneLoggerNeMoCallback."""
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            with patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider:
                mock_instance = MagicMock()
                mock_provider.instance.return_value = mock_instance
                callback = OneLoggerNeMoCallback(mock_instance)

                # TimeEventCallback doesn't use __getattr__, so this should raise AttributeError
                with pytest.raises(AttributeError):
                    _ = callback.test_method

    @pytest.mark.unit
    def test_training_cycle_all_callbacks(self):
        """Test a complete training cycle with all callbacks called in order."""
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            with patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider:
                mock_instance = MagicMock()
                mock_provider.instance.return_value = mock_instance
                callback = OneLoggerNeMoCallback(mock_instance)

                trainer = MagicMock()
                trainer.global_step = 10
                trainer.max_steps = 1000
                pl_module = MagicMock()
                batch = MagicMock()
                outputs = MagicMock()

                # Track all callback calls in order
                callback_calls = []

                with patch('nemo.lightning.one_logger_callback.CB') as mock_cb:
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

    @pytest.mark.unit
    def test_hook_class_init_with_callbacks_inheritance(self):
        """Test hook_class_init_with_callbacks with inheritance to prevent duplicate callbacks."""

        class ParentClass:
            def __init__(self):
                self.parent_initialized = True

        class ChildClass(ParentClass):
            def __init__(self, value=0):
                # Call parent init
                super().__init__()
                self.value = value

        # Hook both classes
        hook_class_init_with_callbacks(ParentClass, "parent_start", "parent_end")
        hook_class_init_with_callbacks(ChildClass, "child_start", "child_end")

        # Create instance and verify callbacks are called correctly
        with patch('nemo.lightning.one_logger_callback.get_one_logger_callbacks') as mock_callbacks:
            instance = ChildClass(42)

            # Should have called both start and end callbacks
            assert mock_callbacks.call_count == 2
            mock_callbacks.assert_any_call("child_start", start_time_msec=patch.ANY)
            mock_callbacks.assert_any_call("child_end", finish_time_msec=patch.ANY)

            # Verify the instance was properly initialized
            assert instance.value == 42
            assert instance.parent_initialized is True

    @pytest.mark.unit
    def test_init_one_logger_no_onelogger(self):
        """Test init_one_logger when OneLogger is not available."""
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', False):
            # Should not raise any exception
            init_one_logger()

    @pytest.mark.unit
    def test_init_one_logger_disabled(self):
        """Test init_one_logger when OneLogger is disabled."""
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            with patch('nemo.lightning.one_logger_callback.enable_onelogger', False):
                # Should not raise any exception
                init_one_logger()

    @pytest.mark.unit
    def test_init_one_logger_already_configured(self):
        """Test init_one_logger when OneLogger is already configured."""
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            with patch('nemo.lightning.one_logger_callback.enable_onelogger', True):
                with patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider:
                    mock_instance = MagicMock()
                    mock_instance.one_logger_ready = True
                    mock_provider.instance.return_value = mock_instance

                    # Should not raise any exception
                    init_one_logger()

    @pytest.mark.unit
    def test_init_one_logger_success(self):
        """Test init_one_logger successful initialization."""
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            with patch('nemo.lightning.one_logger_callback.enable_onelogger', True):
                with patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider:
                    with patch('nemo.lightning.one_logger_callback.get_onelogger_init_config') as mock_get_config:
                        with patch('nemo.lightning.one_logger_callback.OneLoggerConfig') as mock_config_class:
                            with patch('nemo.lightning.one_logger_callback.V1CompatibleExporter') as mock_exporter:
                                with patch(
                                    'nemo.lightning.one_logger_callback.OneLoggerNeMoCallback'
                                ) as mock_callback_class:
                                    mock_instance = MagicMock()
                                    mock_instance.one_logger_ready = False
                                    mock_provider.instance.return_value = mock_instance

                                    mock_config = {"test": "config"}
                                    mock_get_config.return_value = mock_config

                                    mock_config_obj = MagicMock()
                                    mock_config_class.return_value = mock_config_obj

                                    mock_exporter_instance = MagicMock()
                                    mock_exporter.return_value = mock_exporter_instance

                                    mock_callback = MagicMock()
                                    mock_callback_class.return_value = mock_callback

                                    init_one_logger()

                                    # Verify configuration was called
                                    mock_get_config.assert_called_once()
                                    mock_config_class.assert_called_once_with(**mock_config)
                                    mock_exporter.assert_called_once_with(
                                        one_logger_config=mock_config_obj, async_mode=False
                                    )

                                    # Verify provider was configured
                                    mock_instance.with_base_config.assert_called_once_with(mock_config_obj)
                                    mock_instance.with_exporter.assert_called_once_with(
                                        mock_exporter_instance.exporter
                                    )
                                    mock_instance.configure_provider.assert_called_once()

                                    # Verify callback was created
                                    mock_callback_class.assert_called_once_with(mock_instance)

    @pytest.mark.unit
    def test_update_one_logger_config_no_onelogger(self):
        """Test update_one_logger_config when OneLogger is not available."""
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', False):
            trainer = MagicMock()
            # Should not raise any exception
            update_one_logger_config("v1", trainer)

    @pytest.mark.unit
    def test_update_one_logger_config_not_ready(self):
        """Test update_one_logger_config when OneLogger is not ready."""
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            with patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider:
                mock_instance = MagicMock()
                mock_instance.one_logger_ready = False
                mock_provider.instance.return_value = mock_instance

                trainer = MagicMock()
                # Should not raise any exception
                update_one_logger_config("v1", trainer)

    @pytest.mark.unit
    def test_update_one_logger_config_v1(self):
        """Test update_one_logger_config with NeMo v1."""
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            with patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider:
                with patch('nemo.lightning.one_logger_callback.get_nemo_v1_callback_config') as mock_get_config:
                    with patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig') as mock_config_class:
                        with patch('nemo.lightning.one_logger_callback.OneLoggerNeMoCallback') as mock_callback_class:
                            mock_instance = MagicMock()
                            mock_instance.one_logger_ready = True
                            mock_provider.instance.return_value = mock_instance

                            mock_config = {"test": "config"}
                            mock_get_config.return_value = mock_config

                            mock_config_obj = MagicMock()
                            mock_config_class.return_value = mock_config_obj

                            trainer = MagicMock()
                            trainer.callbacks = []

                            mock_callback = MagicMock()
                            mock_callback_class.return_value = mock_callback

                            # Mock the global callback
                            with patch('nemo.lightning.one_logger_callback._ONELOGGER_CALLBACK', mock_callback):
                                update_one_logger_config("v1", trainer)

                                # Verify config was generated and applied
                                mock_get_config.assert_called_once_with(trainer=trainer)
                                mock_config_class.assert_called_once_with(**mock_config)
                                mock_instance.set_training_telemetry_config.assert_called_once_with(mock_config_obj)

                                # Verify callback was added to trainer
                                assert mock_callback in trainer.callbacks

    @pytest.mark.unit
    def test_update_one_logger_config_v2(self):
        """Test update_one_logger_config with NeMo v2."""
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            with patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider:
                with patch('nemo.lightning.one_logger_callback.get_nemo_v2_callback_config') as mock_get_config:
                    with patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig') as mock_config_class:
                        mock_instance = MagicMock()
                        mock_instance.one_logger_ready = True
                        mock_provider.instance.return_value = mock_instance

                        mock_config = {"test": "config"}
                        mock_get_config.return_value = mock_config

                        mock_config_obj = MagicMock()
                        mock_config_class.return_value = mock_config_obj

                        trainer = MagicMock()
                        trainer.callbacks = []

                        # Mock the global callback
                        mock_callback = MagicMock()
                        with patch('nemo.lightning.one_logger_callback._ONELOGGER_CALLBACK', mock_callback):
                            update_one_logger_config("v2", trainer, nemo_logger_config=MagicMock(), data=MagicMock())

                            # Verify config was generated and applied
                            mock_get_config.assert_called_once_with(
                                trainer=trainer, nemo_logger_config=patch.ANY, data=patch.ANY
                            )
                            mock_config_class.assert_called_once_with(**mock_config)
                            mock_instance.set_training_telemetry_config.assert_called_once_with(mock_config_obj)

    @pytest.mark.unit
    def test_update_one_logger_config_invalid_version(self):
        """Test update_one_logger_config with invalid version."""
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            with patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider:
                mock_instance = MagicMock()
                mock_instance.one_logger_ready = True
                mock_provider.instance.return_value = mock_instance

                trainer = MagicMock()

                with pytest.raises(ValueError, match="Invalid NeMo version: invalid"):
                    update_one_logger_config("invalid", trainer)

    @pytest.mark.unit
    def test_update_one_logger_config_callback_already_present(self):
        """Test update_one_logger_config when callback is already in trainer."""
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            with patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider:
                with patch('nemo.lightning.one_logger_callback.get_nemo_v1_callback_config') as mock_get_config:
                    with patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig') as mock_config_class:
                        mock_instance = MagicMock()
                        mock_instance.one_logger_ready = True
                        mock_provider.instance.return_value = mock_instance

                        mock_config = {"test": "config"}
                        mock_get_config.return_value = mock_config

                        mock_config_obj = MagicMock()
                        mock_config_class.return_value = mock_config_obj

                        # Create trainer with existing OneLogger callback
                        trainer = MagicMock()
                        existing_callback = MagicMock()
                        existing_callback.__class__.__name__ = "OneLoggerNeMoCallback"
                        trainer.callbacks = [existing_callback]

                        # Mock the global callback
                        mock_callback = MagicMock()
                        with patch('nemo.lightning.one_logger_callback._ONELOGGER_CALLBACK', mock_callback):
                            update_one_logger_config("v1", trainer)

                            # Verify callback was not added again
                            assert len(trainer.callbacks) == 1
                            assert trainer.callbacks[0] == existing_callback

    @pytest.mark.unit
    def test_update_one_logger_config_no_trainer(self):
        """Test update_one_logger_config with None trainer."""
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            with patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider:
                with patch('nemo.lightning.one_logger_callback.get_nemo_v1_callback_config') as mock_get_config:
                    with patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig') as mock_config_class:
                        mock_instance = MagicMock()
                        mock_instance.one_logger_ready = True
                        mock_provider.instance.return_value = mock_instance

                        mock_config = {"test": "config"}
                        mock_get_config.return_value = mock_config

                        mock_config_obj = MagicMock()
                        mock_config_class.return_value = mock_config_obj

                        # Should not raise any exception
                        update_one_logger_config("v1", None)

                        # Verify config was still generated and applied
                        mock_get_config.assert_called_once_with(trainer=None)
                        mock_config_class.assert_called_once_with(**mock_config)
                        mock_instance.set_training_telemetry_config.assert_called_once_with(mock_config_obj)
