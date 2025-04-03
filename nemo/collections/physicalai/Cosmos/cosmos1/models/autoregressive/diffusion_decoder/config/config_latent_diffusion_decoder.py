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

from typing import Any, List

import attrs

from cosmos1.models.autoregressive.diffusion_decoder.config.registry import register_configs as register_dd_configs
from cosmos1.models.diffusion.config.base.model import LatentDiffusionDecoderModelConfig
from cosmos1.models.diffusion.config.registry import register_configs
from cosmos1.utils import config
from cosmos1.utils.config_helper import import_all_modules_from_package


@attrs.define(slots=False)
class Config(config.Config):
    # default config groups that will be used unless overwritten
    # see config groups in registry.py
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"net": None},
            {"conditioner": "basic"},
            {"tokenizer": "tokenizer"},
            {"tokenizer_corruptor": None},
            {"latent_corruptor": None},
            {"pixel_corruptor": None},
            {"experiment": None},
        ]
    )


def make_config():
    c = Config(model=LatentDiffusionDecoderModelConfig())

    # Specifying values through instances of attrs
    c.job.project = "cosmos_video4"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    # Call this function to register config groups for advanced overriding.
    register_configs()
    register_dd_configs()

    # experiment config are defined in the experiment folder
    # call import_all_modules_from_package to register them
    import_all_modules_from_package("cosmos1.models.diffusion.config.inference", reload=True)
    import_all_modules_from_package("cosmos1.models.autoregressive.diffusion_decoder.config.inference", reload=True)
    return c
