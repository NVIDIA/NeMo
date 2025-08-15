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
    get_current_time_msec,
    get_one_logger_callbacks,
    hook_class_init_with_callbacks,
    init_one_logger,
    update_one_logger_config,
)


# Decorators for cleaner test setup
def with_onelogger_enabled(func):
    """Decorator to enable OneLogger for a test."""

    def wrapper(*args, **kwargs):
        with (
            patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True),
            patch('nemo.lightning.one_logger_callback.enable_onelogger', True),
        ):
            return func(*args, **kwargs)

    return wrapper


def with_onelogger_disabled(func):
    """Decorator to disable OneLogger for a test."""

    def wrapper(*args, **kwargs):
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', False):
            return func(*args, **kwargs)

    return wrapper


def with_onelogger_components(func):
    """Decorator to mock all OneLogger components."""

    def wrapper(*args, **kwargs):
        with (
            patch('nemo.lightning.one_logger_callback.OneLoggerConfig') as mock_config,
            patch('nemo.lightning.one_logger_callback.V1CompatibleExporter') as mock_exporter,
            patch('nemo.lightning.one_logger_callback.OneLoggerNeMoCallback') as mock_callback,
            patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider,
            patch('nemo.lightning.one_logger_callback.get_onelogger_init_config') as mock_get_config,
        ):

            # Setup common mocks
            mock_config_instance = MagicMock()
            mock_config.return_value = mock_config_instance

            mock_provider_instance = MagicMock()
            mock_provider.instance.return_value = mock_provider_instance

            mock_exporter_instance = MagicMock()
            mock_exporter_instance.exporter = MagicMock()
            mock_exporter.return_value = mock_exporter_instance

            mock_callback_instance = MagicMock()
            mock_callback.return_value = mock_callback_instance

            # Setup method chaining
            mock_provider_instance.with_base_config.return_value = mock_provider_instance
            mock_provider_instance.with_exporter.return_value = mock_provider_instance

            return func(
                *args,
                **kwargs,
                mock_config=mock_config,
                mock_config_instance=mock_config_instance,
                mock_exporter=mock_exporter,
                mock_exporter_instance=mock_exporter_instance,
                mock_callback=mock_callback,
                mock_callback_instance=mock_callback_instance,
                mock_provider=mock_provider,
                mock_provider_instance=mock_provider_instance,
                mock_get_config=mock_get_config,
            )

    return wrapper


class TestOneLoggerCallback:
    """Test cases for OneLogger callback functionality."""

    @pytest.mark.unit
    def test_get_current_time_msec(self):
        """Test get_current_time_msec returns current time in milliseconds."""
        time1 = get_current_time_msec()
        time.sleep(0.001)  # Sleep for 1ms
        time2 = get_current_time_msec()

        assert time2 > time1
        assert isinstance(time1, float)
        assert isinstance(time2, float)

    @pytest.mark.unit
    @with_onelogger_disabled
    def test_get_onelogger_callbacks_no_onelogger(self):
        """Test get_one_logger_callbacks when OneLogger is not available."""
        result = get_one_logger_callbacks("test_callback")
        assert result is None

    @pytest.mark.unit
    @with_onelogger_enabled
    def test_get_onelogger_callbacks_with_onelogger(self):
        """Test get_one_logger_callbacks when OneLogger is available."""
        with patch('nemo.lightning.one_logger_callback.CB') as mock_cb:
            mock_callback = MagicMock(return_value="callback_result")
            mock_cb.test_callback = mock_callback

            result = get_one_logger_callbacks("test_callback")
            assert result == "callback_result"
            mock_callback.assert_called_once()

    @pytest.mark.unit
    @with_onelogger_enabled
    def test_get_onelogger_callbacks_attribute_error(self):
        """Test get_one_logger_callbacks handles AttributeError gracefully."""
        with patch('nemo.lightning.one_logger_callback.CB') as mock_cb:
            mock_cb.test_callback = None
            del mock_cb.test_callback

            with pytest.raises(AttributeError, match="OneLogger has no attribute test_callback"):
                get_one_logger_callbacks("test_callback")

    @pytest.mark.unit
    @with_onelogger_enabled
    def test_hook_class_init_with_callbacks_no_init(self):
        """Test hook_class_init_with_callbacks with class that has no __init__ method."""

        class TestClass:
            pass

        hook_class_init_with_callbacks(TestClass, "start_callback", "end_callback")

        assert hasattr(TestClass.__init__, '_one_logger_wrapped')
        assert TestClass.__init__._one_logger_wrapped is True

    @pytest.mark.unit
    @with_onelogger_enabled
    def test_hook_class_init_with_callbacks(self):
        """Test hook_class_init_with_callbacks successfully wraps __init__ method."""

        class TestClass:
            def __init__(self, value=0):
                self.value = value

        original_init = TestClass.__init__
        hook_class_init_with_callbacks(TestClass, "start_callback", "end_callback")

        assert TestClass.__init__ != original_init
        assert hasattr(TestClass.__init__, '_one_logger_wrapped')
        assert TestClass.__init__._one_logger_wrapped is True

    @pytest.mark.unit
    @with_onelogger_enabled
    def test_hook_class_init_with_callbacks_double_wrap(self):
        """Test hook_class_init_with_callbacks handles double wrapping gracefully."""

        class TestClass:
            def __init__(self, value=0):
                self.value = value

        hook_class_init_with_callbacks(TestClass, "start_callback", "end_callback")
        hook_class_init_with_callbacks(TestClass, "start_callback", "end_callback")

        assert hasattr(TestClass.__init__, '_one_logger_wrapped')
        assert TestClass.__init__._one_logger_wrapped is True

    @pytest.mark.unit
    @with_onelogger_enabled
    def test_hook_class_init_with_callbacks_inheritance(self):
        """Test hook_class_init_with_callbacks works with inheritance."""

        class BaseClass:
            def __init__(self, value=0):
                self.value = value

        class DerivedClass(BaseClass):
            def __init__(self, value=0, extra=0):
                super().__init__(value)
                self.extra = extra

        hook_class_init_with_callbacks(DerivedClass, "start_callback", "end_callback")

        assert hasattr(DerivedClass.__init__, '_one_logger_wrapped')
        assert DerivedClass.__init__._one_logger_wrapped is True
        assert not hasattr(BaseClass.__init__, '_one_logger_wrapped')

    @pytest.mark.unit
    @with_onelogger_disabled
    def test_hook_class_init_with_callbacks_no_onelogger(self):
        """Test hook_class_init_with_callbacks when OneLogger is not available."""

        class TestClass:
            def __init__(self, value=0):
                self.value = value

        original_init = TestClass.__init__
        hook_class_init_with_callbacks(TestClass, "start_callback", "end_callback")

        assert TestClass.__init__ == original_init
        assert not hasattr(TestClass.__init__, '_one_logger_wrapped')

    @pytest.mark.unit
    @with_onelogger_disabled
    def test_init_one_logger_no_onelogger(self):
        """Test init_one_logger when OneLogger is not available."""
        init_one_logger()  # Should not raise any exceptions

    @pytest.mark.unit
    @with_onelogger_enabled
    @with_onelogger_components
    def test_init_one_logger_success(
        self,
        mock_get_config,
        mock_config,
        mock_config_instance,
        mock_exporter,
        mock_exporter_instance,
        mock_callback,
        mock_callback_instance,
        mock_provider,
        mock_provider_instance,
    ):
        """Test init_one_logger successful initialization."""
        mock_config_data = {'test': 'data'}
        mock_get_config.return_value = mock_config_data
        mock_provider_instance.one_logger_ready = False

        init_one_logger()

        mock_get_config.assert_called_once()
        mock_config.assert_called_once_with(**mock_config_data)
        mock_exporter.assert_called_once_with(one_logger_config=mock_config_instance, async_mode=False)
        mock_provider_instance.with_base_config.assert_called_once_with(mock_config_instance)
        mock_provider_instance.with_exporter.assert_called_once_with(mock_exporter_instance.exporter)

    @pytest.mark.unit
    @with_onelogger_enabled
    def test_init_one_logger_disabled(self):
        """Test init_one_logger when OneLogger is disabled."""
        with patch('nemo.lightning.one_logger_callback.enable_onelogger', False):
            init_one_logger()  # Should not raise any exceptions

    @pytest.mark.unit
    @with_onelogger_enabled
    def test_init_one_logger_already_configured(self):
        """Test init_one_logger when already configured."""
        with patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider:
            mock_provider_instance = MagicMock()
            mock_provider_instance.one_logger_ready = True
            mock_provider.instance.return_value = mock_provider_instance

            init_one_logger()  # Should not reconfigure

    @pytest.mark.unit
    @with_onelogger_enabled
    def test_update_one_logger_config_nemo_v1(self):
        """Test update_one_logger_config with NeMo v1 configuration."""
        with (
            patch('nemo.lightning.one_logger_callback.get_nemo_v1_callback_config') as mock_get_v1_config,
            patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig') as mock_config_class,
            patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider,
        ):

            mock_config = {'test': 'v1_config'}
            mock_get_v1_config.return_value = mock_config
            mock_config_instance = MagicMock()
            mock_config_class.return_value = mock_config_instance

            mock_provider_instance = MagicMock()
            mock_provider_instance.one_logger_ready = True
            mock_provider.instance.return_value = mock_provider_instance

            mock_trainer = MagicMock()

            update_one_logger_config("v1", mock_trainer, custom_arg="value")

            mock_get_v1_config.assert_called_once_with(trainer=mock_trainer, custom_arg="value")
            mock_config_class.assert_called_once_with(**mock_config)
            mock_provider_instance.set_training_telemetry_config.assert_called_once_with(mock_config_instance)

    @pytest.mark.unit
    @with_onelogger_enabled
    def test_update_one_logger_config_nemo_v2(self):
        """Test update_one_logger_config with NeMo v2 configuration."""
        with (
            patch('nemo.lightning.one_logger_callback.get_nemo_v2_callback_config') as mock_get_v2_config,
            patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig') as mock_config_class,
            patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider,
        ):

            mock_config = {'test': 'v2_config'}
            mock_get_v2_config.return_value = mock_config
            mock_config_instance = MagicMock()
            mock_config_class.return_value = mock_config_instance

            mock_provider_instance = MagicMock()
            mock_provider_instance.one_logger_ready = True
            mock_provider.instance.return_value = mock_provider_instance

            mock_trainer = MagicMock()

            update_one_logger_config("v2", mock_trainer, custom_arg="value")

            mock_get_v2_config.assert_called_once_with(trainer=mock_trainer, custom_arg="value")
            mock_config_class.assert_called_once_with(**mock_config)
            mock_provider_instance.set_training_telemetry_config.assert_called_once_with(mock_config_instance)

    @pytest.mark.unit
    @with_onelogger_disabled
    def test_update_one_logger_config_no_onelogger(self):
        """Test update_one_logger_config when OneLogger is not available."""
        update_one_logger_config("v1", MagicMock())  # Should not raise any exceptions

    @pytest.mark.unit
    @with_onelogger_enabled
    def test_update_one_logger_config_disabled(self):
        """Test update_one_logger_config when OneLogger is disabled."""
        with (
            patch('nemo.lightning.one_logger_callback.enable_onelogger', False),
            patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider,
            patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig') as mock_config_class,
        ):

            mock_provider_instance = MagicMock()
            mock_provider_instance.one_logger_ready = True
            mock_provider.instance.return_value = mock_provider_instance

            # Create a properly mocked trainer
            mock_trainer = MagicMock()
            mock_trainer.max_steps = 100
            mock_trainer.log_every_n_steps = 10
            mock_trainer.val_check_interval = 0
            mock_trainer.callbacks = []
            mock_trainer.strategy = None

            # Mock the lightning module config
            mock_lightning_module = MagicMock()
            mock_cfg = MagicMock()
            mock_train_ds = MagicMock()
            mock_train_ds.batch_size = 32
            mock_cfg.train_ds = mock_train_ds
            mock_encoder = MagicMock()
            mock_encoder.d_model = 512
            mock_cfg.encoder = mock_encoder
            mock_lightning_module.cfg = mock_cfg
            mock_trainer.lightning_module = mock_lightning_module

            # Mock the config class
            mock_config_instance = MagicMock()
            mock_config_class.return_value = mock_config_instance

            update_one_logger_config("v1", mock_trainer)  # Should not raise any exceptions

    @pytest.mark.unit
    @with_onelogger_enabled
    def test_update_one_logger_config_invalid_version(self):
        """Test update_one_logger_config with invalid version."""
        with (
            patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider,
            patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig') as mock_config_class,
        ):

            mock_provider_instance = MagicMock()
            mock_provider_instance.one_logger_ready = True
            mock_provider.instance.return_value = mock_provider_instance

            # Create a properly mocked trainer
            mock_trainer = MagicMock()
            mock_trainer.max_steps = 100
            mock_trainer.log_every_n_steps = 10
            mock_trainer.val_check_interval = 0
            mock_trainer.callbacks = []
            mock_trainer.strategy = None

            # Mock the lightning module config
            mock_lightning_module = MagicMock()
            mock_cfg = MagicMock()
            mock_train_ds = MagicMock()
            mock_train_ds.batch_size = 32
            mock_cfg.train_ds = mock_train_ds
            mock_encoder = MagicMock()
            mock_encoder.d_model = 512
            mock_cfg.encoder = mock_encoder
            mock_lightning_module.cfg = mock_cfg
            mock_trainer.lightning_module = mock_lightning_module

            # Mock the config class
            mock_config_instance = MagicMock()
            mock_config_class.return_value = mock_config_instance

            # Should fall back to v1 without errors
            update_one_logger_config("invalid", mock_trainer)

    @pytest.mark.unit
    @with_onelogger_enabled
    def test_update_one_logger_config_exception_handling(self):
        """Test update_one_logger_config exception handling."""
        with (
            patch('nemo.lightning.one_logger_callback.get_nemo_v1_callback_config') as mock_get_config,
            patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider,
        ):

            mock_provider_instance = MagicMock()
            mock_provider_instance.one_logger_ready = True
            mock_provider.instance.return_value = mock_provider_instance

            mock_get_config.side_effect = Exception("Configuration error")

            # Should raise the exception (no more try-catch)
            with pytest.raises(Exception, match="Configuration error"):
                update_one_logger_config("v1", MagicMock())

    @pytest.mark.unit
    @with_onelogger_enabled
    @with_onelogger_components
    def test_init_one_logger_world_size_in_init_config(
        self,
        mock_get_config,
        mock_config,
        mock_config_instance,
        mock_exporter,
        mock_exporter_instance,
        mock_callback,
        mock_callback_instance,
        mock_provider,
        mock_provider_instance,
    ):
        """Test that world_size is included in the init config, not the callback config."""
        mock_config_data = {'world_size_or_fn': 8, 'application_name': 'test'}
        mock_get_config.return_value = mock_config_data
        mock_provider_instance.one_logger_ready = False

        init_one_logger()

        mock_get_config.assert_called_once()
        mock_config.assert_called_once_with(**mock_config_data)

        assert 'world_size_or_fn' in mock_config_data
        assert mock_config_data['world_size_or_fn'] == 8
