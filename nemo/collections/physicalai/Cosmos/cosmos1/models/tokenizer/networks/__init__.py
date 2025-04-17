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

from enum import Enum

from cosmos1.models.tokenizer.networks.configs import continuous_image_8x8 as continuous_image_8x8_dict
from cosmos1.models.tokenizer.networks.configs import continuous_image_8x8_lowres as continuous_image_8x8_lowres_dict
from cosmos1.models.tokenizer.networks.configs import continuous_image_16x16 as continuous_image_16x16_dict
from cosmos1.models.tokenizer.networks.configs import (
    continuous_image_16x16_lowres as continuous_image_16x16_lowres_dict,
)
from cosmos1.models.tokenizer.networks.configs import continuous_video_4x8x8 as continuous_video_4x8x8_dict
from cosmos1.models.tokenizer.networks.configs import (
    continuous_video_4x8x8_lowres as continuous_video_4x8x8_lowres_dict,
)
from cosmos1.models.tokenizer.networks.configs import continuous_video_8x8x8 as continuous_video_8x8x8_dict
from cosmos1.models.tokenizer.networks.configs import continuous_video_8x16x16 as continuous_video_8x16x16_dict
from cosmos1.models.tokenizer.networks.configs import discrete_image_8x8 as discrete_image_8x8_dict
from cosmos1.models.tokenizer.networks.configs import discrete_image_8x8_lowres as discrete_image_8x8_lowres_dict
from cosmos1.models.tokenizer.networks.configs import discrete_image_16x16 as cdiscrete_image_16x16_dict
from cosmos1.models.tokenizer.networks.configs import discrete_image_16x16_lowres as discrete_image_16x16_lowres_dict
from cosmos1.models.tokenizer.networks.configs import discrete_video_4x8x8 as discrete_video_4x8x8_dict
from cosmos1.models.tokenizer.networks.configs import discrete_video_4x8x8_lowres as discrete_video_4x8x8_lowres_dict
from cosmos1.models.tokenizer.networks.configs import discrete_video_8x8x8 as discrete_video_8x8x8_dict
from cosmos1.models.tokenizer.networks.configs import discrete_video_8x16x16 as discrete_video_8x16x16_dict
from cosmos1.models.tokenizer.networks.continuous_image import ContinuousImageTokenizer
from cosmos1.models.tokenizer.networks.continuous_video import CausalContinuousVideoTokenizer
from cosmos1.models.tokenizer.networks.discrete_image import DiscreteImageTokenizer
from cosmos1.models.tokenizer.networks.discrete_video import CausalDiscreteVideoTokenizer


class TokenizerConfigs(Enum):
    """Continuous Image (CI) Tokenizer Configs"""

    # Cosmos-0.1-Tokenizer-CI8x8
    CI8x8 = continuous_image_8x8_dict

    # Cosmos-0.1-Tokenizer-CI16x16
    CI16x16 = continuous_image_16x16_dict

    # Cosmos-1.0-Tokenizer-CI8x8-LowRes
    CI8x8_LowRes = continuous_image_8x8_lowres_dict

    # Cosmos-1.0-Tokenizer-CI16x16-LowRes
    CI16x16_LowRes = continuous_image_16x16_lowres_dict

    """Discrete Image (DI) Tokenizer Configs"""
    # Cosmos-0.1-Tokenizer-DI8x8
    DI8x8 = discrete_image_8x8_dict

    # Cosmos-0.1-Tokenizer-DI16x16
    DI16x16 = cdiscrete_image_16x16_dict

    # Cosmos-1.0-Tokenizer-DI8x8-LowRes
    DI8x8_LowRes = discrete_image_8x8_lowres_dict

    # Cosmos-1.0-Tokenizer-DI16x16-LowRes
    DI16x16_LowRes = discrete_image_16x16_lowres_dict

    """Causal Continuous Video (CV) Tokenizer Configs"""
    # Cosmos-0.1-Tokenizer-CV4x8x8
    CV4x8x8 = continuous_video_4x8x8_dict

    # Cosmos-0.1-Tokenizer-CV8x8x8 and Cosmos-1.0-Tokenizer-CV8x8x8
    CV8x8x8 = continuous_video_8x8x8_dict

    # Cosmos-0.1-Tokenizer-CV8x16x16
    CV8x16x16 = continuous_video_8x16x16_dict

    # Cosmos-1.0-Tokenizer-CV4x8x8-LowRes
    CV4x8x8_LowRes = continuous_video_4x8x8_lowres_dict

    """Causal Discrete Video (DV) Tokenizer Configs"""
    # Cosmos-0.1-Tokenizer-DV4x8x8
    DV4x8x8 = discrete_video_4x8x8_dict

    # Cosmos-0.1-Tokenizer-DV8x8x8
    DV8x8x8 = discrete_video_8x8x8_dict

    # Cosmos-0.1-Tokenizer-DV8x16x16 and Cosmos-1.0-Tokenizer-DV8x16x16
    DV8x16x16 = discrete_video_8x16x16_dict

    # Cosmos-1.0-Tokenizer-DV4x8x8-LowRes
    DV4x8x8_LowRes = discrete_video_4x8x8_lowres_dict


class TokenizerModels(Enum):
    CI = ContinuousImageTokenizer
    DI = DiscreteImageTokenizer
    CV = CausalContinuousVideoTokenizer
    DV = CausalDiscreteVideoTokenizer
