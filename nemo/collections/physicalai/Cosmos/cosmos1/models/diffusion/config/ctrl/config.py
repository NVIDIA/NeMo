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

from typing import Any, List

import attrs

from cosmos1.models.diffusion.config.ctrl.model import CtrlModelConfig

# from cosmos1.models.diffusion.config.base.model import DefaultModelConfig
from cosmos1.models.diffusion.config.ctrl.registry import register_configs
from cosmos1.models.diffusion.model.model_ctrl import VideoDiffusionModelWithCtrl
from cosmos1.utils import config
from cosmos1.utils.config_helper import import_all_modules_from_package
from cosmos1.utils.lazy_config import PLACEHOLDER
from cosmos1.utils.lazy_config import LazyCall as L
from cosmos1.utils.lazy_config import LazyDict

# import cosmos1.models.diffusion.config.config as base_config
# from cosmos1.models.diffusion.checkpointers.ema_fsdp_checkpointer import CheckpointConfig


@attrs.define(slots=False)
class Config(config.Config):
    # default config groups that will be used unless overwritten
    # see config groups in registry.py
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"net": None},
            {"net_ctrl": None},
            {"hint_key": "control_input_canny"},
            {"conditioner": "ctrlnet_add_fps_image_size_padding_mask"},
            {"tokenizer": "vae1"},
            {"experiment": None},
        ]
    )
    model_obj: LazyDict = L(VideoDiffusionModelWithCtrl)(
        config=PLACEHOLDER,
    )


def make_config():
    c = Config(
        model=CtrlModelConfig(),
    )
    # Specifying values through instances of attrs

    # Call this function to register config groups for advanced overriding.
    register_configs()

    # experiment config are defined in the experiment folder
    # call import_all_modules_from_package to register them
    import_all_modules_from_package("cosmos1.models.diffusion.config.inference.ctrl")
    return c
