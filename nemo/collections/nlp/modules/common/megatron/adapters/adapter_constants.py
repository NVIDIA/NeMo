# coding=utf-8
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

from dataclasses import dataclass

from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    InfusedAdapterConfig,
    ParallelLinearAdapterConfig,
)
from nemo.core.classes.mixins.adapter_mixins import AdapterSpec


@dataclass
class AdapterType:
    # Adapters requires in the IA3 schema
    KEY_INFUSED: AdapterSpec = AdapterSpec(name="key_infused_adapter", targets=[InfusedAdapterConfig._target_])
    VALUE_INFUSED: AdapterSpec = AdapterSpec(name="value_infused_adapter", targets=[InfusedAdapterConfig._target_])
    MLP_INFUSED: AdapterSpec = AdapterSpec(name="mlp_infused_adapter", targets=[InfusedAdapterConfig._target_])

    # Standard Adapters
    ADAPTER_ONE: AdapterSpec = AdapterSpec(
        name="adapter_1", targets=[ParallelLinearAdapterConfig._target_, LinearAdapterConfig._target_]
    )
    ADAPTER_TWO: AdapterSpec = AdapterSpec(
        name="adapter_2", targets=[ParallelLinearAdapterConfig._target_, LinearAdapterConfig._target_]
    )
