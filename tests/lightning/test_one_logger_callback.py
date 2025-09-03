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

"""Unit tests for OneLoggerNeMoCallback."""

import os
from unittest.mock import MagicMock, patch

import pytest
from lightning.pytorch.callbacks import Callback as PTLCallback

from nemo.lightning.base_callback import BaseCallback
from nemo.lightning.one_logger_callback import OneLoggerNeMoCallback


class TestOneLoggerNeMoCallback:
    """Test suite for OneLoggerNeMoCallback."""

    def test_inheritance(self):
        """Test that OneLoggerNeMoCallback properly inherits from both parent classes."""
        with (
            patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback') as mock_ptl_callback,
            patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider,
            patch('nemo.lightning.one_logger_callback.get_one_logger_init_config') as mock_get_config,
            patch('nemo.lightning.one_logger_callback.OneLoggerConfig') as mock_config_class,
        ):

            # Setup mocks
            mock_get_config.return_value = {"application_name": "test", "session_tag_or_fn": "test-session"}
            mock_config_instance = MagicMock()
            mock_config_class.return_value = mock_config_instance
            mock_provider_instance = MagicMock()
            mock_provider.instance.return_value = mock_provider_instance
            mock_ptl_callback_instance = MagicMock()
            mock_ptl_callback.return_value = mock_ptl_callback_instance

            # Create callback instance
            callback = OneLoggerNeMoCallback()

            # Test inheritance
            assert isinstance(callback, BaseCallback)
            assert isinstance(callback, PTLCallback)

    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback.__init__', return_value=None)
    def test_init_configures_provider(self, mock_ptl_callback_init, mock_config_class, mock_get_config, mock_provider):
        """Test that __init__ properly configures the OneLogger provider."""
        # Setup mocks
        mock_init_config = {
            "application_name": "nemo",
            "session_tag_or_fn": "test-session",
            "enable_for_current_rank": True,
            "world_size_or_fn": 1,
            "error_handling_strategy": "propagate_exceptions",
        }
        mock_get_config.return_value = mock_init_config

        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance

        mock_provider_instance = MagicMock()
        mock_provider.instance.return_value = mock_provider_instance

        # Create callback instance
        callback = OneLoggerNeMoCallback()

        # Verify initialization sequence
        mock_get_config.assert_called_once()
        mock_config_class.assert_called_once_with(**mock_init_config)
        mock_provider_instance.with_base_config.assert_called_once_with(mock_config_instance)
        mock_provider_instance.with_base_config.return_value.with_export_config.assert_called_once()
        mock_provider_instance.with_base_config.return_value.with_export_config.return_value.configure_provider.assert_called_once()
        mock_ptl_callback_init.assert_called_once_with(mock_provider_instance)

    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_nemo_v1_callback_config')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback')
    def test_update_config_v1(
        self,
        mock_ptl_callback,
        mock_config_class,
        mock_get_config,
        mock_telemetry_config_class,
        mock_get_v1_config,
        mock_provider,
    ):
        """Test update_config with nemo_version='v1'."""
        # Setup mocks
        mock_get_config.return_value = {"application_name": "test"}
        mock_config_class.return_value = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider.instance.return_value = mock_provider_instance
        mock_ptl_callback.return_value = MagicMock()

        mock_v1_config = {"job_name": "test-job", "world_size": 1, "global_batch_size": 32, "seq_length": 1024}
        mock_get_v1_config.return_value = mock_v1_config

        mock_telemetry_config_instance = MagicMock()
        mock_telemetry_config_class.return_value = mock_telemetry_config_instance

        # Create callback and trainer
        callback = OneLoggerNeMoCallback()
        trainer = MagicMock()

        # Call update_config
        callback.update_config(nemo_version='v1', trainer=trainer)

        # Verify v1 config was called
        mock_get_v1_config.assert_called_once_with(trainer=trainer)
        mock_telemetry_config_class.assert_called_once_with(**mock_v1_config)
        mock_provider_instance.set_training_telemetry_config.assert_called_once_with(mock_telemetry_config_instance)

    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_nemo_v2_callback_config')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback')
    def test_update_config_v2(
        self,
        mock_ptl_callback,
        mock_config_class,
        mock_get_config,
        mock_telemetry_config_class,
        mock_get_v2_config,
        mock_provider,
    ):
        """Test update_config with nemo_version='v2' and data module."""
        # Setup mocks
        mock_get_config.return_value = {"application_name": "test"}
        mock_config_class.return_value = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider.instance.return_value = mock_provider_instance
        mock_ptl_callback.return_value = MagicMock()

        mock_v2_config = {"job_name": "test-job-v2", "world_size": 2, "global_batch_size": 64, "seq_length": 2048}
        mock_get_v2_config.return_value = mock_v2_config

        mock_telemetry_config_instance = MagicMock()
        mock_telemetry_config_class.return_value = mock_telemetry_config_instance

        # Create callback, trainer, and data module
        callback = OneLoggerNeMoCallback()
        trainer = MagicMock()
        data_module = MagicMock()

        # Call update_config with v2 and data
        callback.update_config(nemo_version='v2', trainer=trainer, data=data_module)

        # Verify v2 config was called with data
        mock_get_v2_config.assert_called_once_with(trainer=trainer, data=data_module)
        mock_telemetry_config_class.assert_called_once_with(**mock_v2_config)
        mock_provider_instance.set_training_telemetry_config.assert_called_once_with(mock_telemetry_config_instance)

    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_nemo_v1_callback_config')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback')
    def test_update_config_unknown_version_defaults_to_v1(
        self,
        mock_ptl_callback,
        mock_config_class,
        mock_get_config,
        mock_telemetry_config_class,
        mock_get_v1_config,
        mock_provider,
    ):
        """Test update_config with unknown version defaults to v1."""
        # Setup mocks
        mock_get_config.return_value = {"application_name": "test"}
        mock_config_class.return_value = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider.instance.return_value = mock_provider_instance
        mock_ptl_callback.return_value = MagicMock()

        mock_v1_config = {"job_name": "test-job"}
        mock_get_v1_config.return_value = mock_v1_config

        mock_telemetry_config_instance = MagicMock()
        mock_telemetry_config_class.return_value = mock_telemetry_config_instance

        # Create callback and trainer
        callback = OneLoggerNeMoCallback()
        trainer = MagicMock()

        # Call update_config with unknown version
        callback.update_config(nemo_version='unknown', trainer=trainer)

        # Verify v1 config was called (default fallback)
        mock_get_v1_config.assert_called_once_with(trainer=trainer)
        mock_telemetry_config_class.assert_called_once_with(**mock_v1_config)
        mock_provider_instance.set_training_telemetry_config.assert_called_once_with(mock_telemetry_config_instance)

    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_nemo_v2_callback_config')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback')
    def test_update_config_v2_without_data(
        self,
        mock_ptl_callback,
        mock_config_class,
        mock_get_config,
        mock_telemetry_config_class,
        mock_get_v2_config,
        mock_provider,
    ):
        """Test update_config with nemo_version='v2' but no data module provided."""
        # Setup mocks
        mock_get_config.return_value = {"application_name": "test"}
        mock_config_class.return_value = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider.instance.return_value = mock_provider_instance
        mock_ptl_callback.return_value = MagicMock()

        mock_v2_config = {"job_name": "test-job-v2"}
        mock_get_v2_config.return_value = mock_v2_config

        mock_telemetry_config_instance = MagicMock()
        mock_telemetry_config_class.return_value = mock_telemetry_config_instance

        # Create callback and trainer
        callback = OneLoggerNeMoCallback()
        trainer = MagicMock()

        # Call update_config with v2 but no data
        callback.update_config(nemo_version='v2', trainer=trainer)

        # Verify v2 config was called with None data
        mock_get_v2_config.assert_called_once_with(trainer=trainer, data=None)
        mock_telemetry_config_class.assert_called_once_with(**mock_v2_config)
        mock_provider_instance.set_training_telemetry_config.assert_called_once_with(mock_telemetry_config_instance)

    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_nemo_v2_callback_config')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback')
    def test_update_config_v2_with_extra_kwargs(
        self,
        mock_ptl_callback,
        mock_config_class,
        mock_get_config,
        mock_telemetry_config_class,
        mock_get_v2_config,
        mock_provider,
    ):
        """Test update_config with nemo_version='v2' and extra kwargs."""
        # Setup mocks
        mock_get_config.return_value = {"application_name": "test"}
        mock_config_class.return_value = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider.instance.return_value = mock_provider_instance
        mock_ptl_callback.return_value = MagicMock()

        mock_v2_config = {"job_name": "test-job-v2"}
        mock_get_v2_config.return_value = mock_v2_config

        mock_telemetry_config_instance = MagicMock()
        mock_telemetry_config_class.return_value = mock_telemetry_config_instance

        # Create callback and trainer
        callback = OneLoggerNeMoCallback()
        trainer = MagicMock()
        data_module = MagicMock()

        # Call update_config with v2, data, and extra kwargs
        callback.update_config(
            nemo_version='v2', trainer=trainer, data=data_module, extra_param1='value1', extra_param2='value2'
        )

        # Verify v2 config was called with data (extra kwargs should be ignored)
        mock_get_v2_config.assert_called_once_with(trainer=trainer, data=data_module)
        mock_telemetry_config_class.assert_called_once_with(**mock_v2_config)
        mock_provider_instance.set_training_telemetry_config.assert_called_once_with(mock_telemetry_config_instance)

    def test_export_all_symbols(self):
        """Test that __all__ contains the expected symbols."""
        from nemo.lightning.one_logger_callback import __all__

        assert 'OneLoggerNeMoCallback' in __all__

    @patch.dict(os.environ, {'EXP_NAME': 'test-experiment', 'WORLD_SIZE': '4'})
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback')
    def test_init_with_environment_variables(
        self, mock_ptl_callback, mock_config_class, mock_get_config, mock_provider
    ):
        """Test initialization with environment variables set."""
        # Setup mocks
        mock_get_config.return_value = {
            "application_name": "nemo",
            "session_tag_or_fn": "test-experiment",
            "world_size_or_fn": 4,
        }
        mock_config_class.return_value = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider.instance.return_value = mock_provider_instance
        mock_ptl_callback.return_value = MagicMock()

        # Create callback instance
        callback = OneLoggerNeMoCallback()

        # Verify that get_one_logger_init_config was called
        mock_get_config.assert_called_once()

        # Verify that the config was created with the environment-based values
        mock_config_class.assert_called_once()
        call_args = mock_config_class.call_args[1]
        assert call_args['session_tag_or_fn'] == 'test-experiment'
        assert call_args['world_size_or_fn'] == 4

    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_nemo_v1_callback_config')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback')
    def test_update_config_with_empty_config(
        self,
        mock_ptl_callback,
        mock_config_class,
        mock_get_config,
        mock_telemetry_config_class,
        mock_get_v1_config,
        mock_provider,
    ):
        """Test update_config with empty configuration dictionary."""
        # Setup mocks
        mock_get_config.return_value = {"application_name": "test"}
        mock_config_class.return_value = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider.instance.return_value = mock_provider_instance
        mock_ptl_callback.return_value = MagicMock()

        # Return empty config
        mock_get_v1_config.return_value = {}

        mock_telemetry_config_instance = MagicMock()
        mock_telemetry_config_class.return_value = mock_telemetry_config_instance

        # Create callback and trainer
        callback = OneLoggerNeMoCallback()
        trainer = MagicMock()

        # Call update_config
        callback.update_config(nemo_version='v1', trainer=trainer)

        # Verify empty config was passed to TrainingTelemetryConfig
        mock_telemetry_config_class.assert_called_once_with(**{})
        mock_provider_instance.set_training_telemetry_config.assert_called_once_with(mock_telemetry_config_instance)

    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_nemo_v1_callback_config')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback')
    def test_update_config_provider_exception_handling(
        self,
        mock_ptl_callback,
        mock_config_class,
        mock_get_config,
        mock_telemetry_config_class,
        mock_get_v1_config,
        mock_provider,
    ):
        """Test update_config handles provider exceptions gracefully."""
        # Setup mocks
        mock_get_config.return_value = {"application_name": "test"}
        mock_config_class.return_value = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider.instance.return_value = mock_provider_instance
        mock_ptl_callback.return_value = MagicMock()

        mock_v1_config = {"job_name": "test-job"}
        mock_get_v1_config.return_value = mock_v1_config

        mock_telemetry_config_instance = MagicMock()
        mock_telemetry_config_class.return_value = mock_telemetry_config_instance

        # Make provider raise an exception
        mock_provider_instance.set_training_telemetry_config.side_effect = Exception("Provider error")

        # Create callback and trainer
        callback = OneLoggerNeMoCallback()
        trainer = MagicMock()

        # Call update_config and expect exception to be raised
        with pytest.raises(Exception, match="Provider error"):
            callback.update_config(nemo_version='v1', trainer=trainer)

    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_nemo_v1_callback_config')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback')
    def test_update_config_config_exception_handling(
        self,
        mock_ptl_callback,
        mock_config_class,
        mock_get_config,
        mock_telemetry_config_class,
        mock_get_v1_config,
        mock_provider,
    ):
        """Test update_config handles TrainingTelemetryConfig exceptions gracefully."""
        # Setup mocks
        mock_get_config.return_value = {"application_name": "test"}
        mock_config_class.return_value = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider.instance.return_value = mock_provider_instance
        mock_ptl_callback.return_value = MagicMock()

        mock_v1_config = {"job_name": "test-job"}
        mock_get_v1_config.return_value = mock_v1_config

        # Make TrainingTelemetryConfig raise an exception
        mock_telemetry_config_class.side_effect = Exception("Config error")

        # Create callback and trainer
        callback = OneLoggerNeMoCallback()
        trainer = MagicMock()

        # Call update_config and expect exception to be raised
        with pytest.raises(Exception, match="Config error"):
            callback.update_config(nemo_version='v1', trainer=trainer)

    def test_callback_instantiation_without_mocks_raises_import_error(self):
        """Test that callback instantiation without proper mocks raises appropriate errors."""
        # This test verifies that the callback properly depends on external libraries
        # and will raise import errors if they're not available
        with patch(
            'nemo.lightning.one_logger_callback.OneLoggerPTLCallback',
            side_effect=Exception("with_base_config can be called only before configure_provider is called."),
        ):
            with pytest.raises(
                Exception, match="with_base_config can be called only before configure_provider is called."
            ):
                OneLoggerNeMoCallback()

    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback.__init__', return_value=None)
    def test_init_provider_chain_calls(
        self, mock_ptl_callback_init, mock_config_class, mock_get_config, mock_provider
    ):
        """Test that the provider configuration chain is called in correct order."""
        # Setup mocks
        mock_get_config.return_value = {"application_name": "test"}
        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance
        mock_provider_instance = MagicMock()
        mock_provider.instance.return_value = mock_provider_instance

        # Create callback instance
        OneLoggerNeMoCallback()

        # Verify the provider configuration chain
        mock_provider_instance.with_base_config.assert_called_once_with(mock_config_instance)
        chain_result = mock_provider_instance.with_base_config.return_value
        chain_result.with_export_config.assert_called_once()
        chain_result.with_export_config.return_value.configure_provider.assert_called_once()

        # Verify PTL callback was initialized with provider instance
        mock_ptl_callback_init.assert_called_once_with(mock_provider_instance)
