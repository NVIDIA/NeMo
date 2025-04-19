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

from typing import List

import attrs

from cosmos1.utils.lazy_config import LazyDict


@attrs.define(slots=False)
class DefaultModelConfig:
    tokenizer: LazyDict = None
    conditioner: LazyDict = None
    net: LazyDict = None
    sigma_data: float = 0.5
    precision: str = "bfloat16"
    input_data_key: str = "video"  # key to fetch input data from data_batch
    latent_shape: List[int] = [16, 24, 44, 80]  # 24 corresponig to 136 frames


@attrs.define(slots=False)
class LatentDiffusionDecoderModelConfig(DefaultModelConfig):
    tokenizer_corruptor: LazyDict = None
    latent_corruptor: LazyDict = None
    pixel_corruptor: LazyDict = None
    diffusion_decoder_cond_sigma_low: float = None
    diffusion_decoder_cond_sigma_high: float = None
    diffusion_decoder_corrupt_prob: float = None
    condition_on_tokenizer_corruptor_token: bool = False


@attrs.define(slots=False)
class MultiCameraConfig(DefaultModelConfig):
    n_cameras: int = 4
