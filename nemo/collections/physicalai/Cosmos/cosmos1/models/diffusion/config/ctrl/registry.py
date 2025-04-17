# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0115,C0116,C0301

import cosmos1.models.diffusion.config.registry as base_registry
from cosmos1.models.diffusion.config.ctrl.conditioner import (
    CTRL_HINT_KEYS,
    BaseVideoConditionerWithCtrlConfig,
    VideoConditionerFpsSizePaddingWithCtrlConfig,
)
from cosmos1.models.diffusion.config.ctrl.net_ctrl import FADITV2_14B_EncoderConfig, FADITV2EncoderConfig
from hydra.core.config_store import ConfigStore


def register_experiment_ctrlnet(cs):
    cs.store(group="net_ctrl", package="model.net_ctrl", name="faditv2_7b", node=FADITV2EncoderConfig)
    cs.store(group="net_ctrl", package="model.net_ctrl", name="faditv2_14b", node=FADITV2_14B_EncoderConfig)

    cs.store(group="conditioner", package="model.conditioner", name="ctrlnet", node=BaseVideoConditionerWithCtrlConfig)
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="ctrlnet_add_fps_image_size_padding_mask",
        node=VideoConditionerFpsSizePaddingWithCtrlConfig,
    )
    for hint_key in CTRL_HINT_KEYS:
        cs.store(
            group="hint_key",
            package="model",
            name=hint_key,
            node=dict(hint_key=dict(hint_key=hint_key, grayscale=False)),
        )
        cs.store(
            group="hint_key",
            package="model",
            name=f"{hint_key}_grayscale",
            node=dict(hint_key=dict(hint_key=hint_key, grayscale=True)),
        )


def register_configs():
    cs = ConfigStore.instance()
    base_registry.register_configs()
    register_experiment_ctrlnet(cs)
