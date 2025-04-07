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

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from cosmos1.models.diffusion.conditioner import BaseVideoCondition, GeneralConditioner
from cosmos1.models.diffusion.config.base.conditioner import (
    FPSConfig,
    ImageSizeConfig,
    LatentConditionConfig,
    LatentConditionSigmaConfig,
    NumFramesConfig,
    PaddingMaskConfig,
    TextConfig,
)
from cosmos1.utils.lazy_config import LazyCall as L
from cosmos1.utils.lazy_config import LazyDict


@dataclass
class VideoLatentDiffusionDecoderCondition(BaseVideoCondition):
    # latent_condition will concat to the input of network, along channel dim;
    # cfg will make latent_condition all zero padding.
    latent_condition: Optional[torch.Tensor] = None
    latent_condition_sigma: Optional[torch.Tensor] = None


class VideoDiffusionDecoderConditioner(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> VideoLatentDiffusionDecoderCondition:
        output = super()._forward(batch, override_dropout_rate)
        return VideoLatentDiffusionDecoderCondition(**output)


VideoLatentDiffusionDecoderConditionerConfig: LazyDict = L(VideoDiffusionDecoderConditioner)(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
    image_size=ImageSizeConfig(),
    padding_mask=PaddingMaskConfig(),
    latent_condition=LatentConditionConfig(),
    latent_condition_sigma=LatentConditionSigmaConfig(),
)
