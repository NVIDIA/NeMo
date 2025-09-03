# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""
OneLogger callback for NeMo training.

This module provides a callback that integrates OneLogger telemetry with NeMo training.
"""
from lightning.pytorch import Trainer
from nv_one_logger.api.config import OneLoggerConfig
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider
from nv_one_logger.training_telemetry.integration.pytorch_lightning import TimeEventCallback as OneLoggerPTLCallback

from nemo.lightning.base_callback import BaseCallback
from nemo.utils.meta_info_manager import (
    enable_one_logger,
    get_nemo_v1_callback_config,
    get_nemo_v2_callback_config,
    get_one_logger_init_config,
)

# Export all symbols for testing and usage
__all__ = ['OneLoggerNeMoCallback']


class OneLoggerNeMoCallback(OneLoggerPTLCallback, BaseCallback):
    """Adapter extending OneLogger's PTL callback with init + config update.

    __init__ configures the provider from meta info, then calls super().__init__.
    update_config computes TrainingTelemetryConfig and applies it.
    """

    def __init__(self) -> None:
        # Configure provider from meta info
        init_config = get_one_logger_init_config()
        one_logger_config = OneLoggerConfig(**init_config)
        TrainingTelemetryProvider.instance().with_base_config(
            one_logger_config
        ).with_export_config().configure_provider()
        # Initialize underlying OneLogger PTL callback
        super().__init__(TrainingTelemetryProvider.instance())

    def update_config(self, nemo_version: str, trainer: Trainer, **kwargs) -> None:
        if nemo_version == 'v1':
            config = get_nemo_v1_callback_config(trainer=trainer)
        elif nemo_version == 'v2':
            # v2 expects data module in kwargs
            data = kwargs.get('data', None)
            config = get_nemo_v2_callback_config(trainer=trainer, data=data)
        else:
            config = get_nemo_v1_callback_config(trainer=trainer)
        training_telemetry_config = TrainingTelemetryConfig(**config)
        TrainingTelemetryProvider.instance().set_training_telemetry_config(training_telemetry_config)
