# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from enum import Enum
from nemo.utils.enum import PrettyStrEnum
import torch


class LocalTransformerType(PrettyStrEnum):
    """
    Enum for the type of local transformer to use in the MagpieTTS model.
    These strings are the values allowed in the YAML config file.
    """

    NO_LT = "none"
    AR = "autoregressive"
    MASKGIT = "maskgit"


class SpecialAudioToken(Enum):
    """
    Enum for the special tokens to use in the MagpieTTS model.
    The special tokens are appended at the end of the codebook after the actual audio codec tokens.
    The actual codeco index is this value below plus the number of codec tokens - do not use the Enum directly.
    """

    AUDIO_BOS = 0
    AUDIO_EOS = 1
    AUDIO_CONTEXT_BOS = 2
    AUDIO_CONTEXT_EOS = 3
    MASK_TOKEN = 4
    # Reserve these values so that if we need to add more special tokens in the future the codebook size will remain the same
    RESERVED_1 = 5
    RESERVED_2 = 6
    RESERVED_3 = 7


def cosine_schedule(x: torch.Tensor):
    """
    Maps input values from [0, 1] to [1, 0] using the first quadrant of the cosine function.
    Used for MaskGit mask scheduling.
    """
    return torch.cos(x * (torch.pi / 2))
