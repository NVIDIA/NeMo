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

from cosmos1.models.tokenizer.modules.distributions import GaussianDistribution, IdentityDistribution
from cosmos1.models.tokenizer.modules.layers2d import Decoder, Encoder
from cosmos1.models.tokenizer.modules.layers3d import DecoderBase, DecoderFactorized, EncoderBase, EncoderFactorized
from cosmos1.models.tokenizer.modules.quantizers import FSQuantizer, LFQuantizer, ResidualFSQuantizer, VectorQuantizer


class EncoderType(Enum):
    Default = Encoder


class DecoderType(Enum):
    Default = Decoder


class Encoder3DType(Enum):
    BASE = EncoderBase
    FACTORIZED = EncoderFactorized


class Decoder3DType(Enum):
    BASE = DecoderBase
    FACTORIZED = DecoderFactorized


class ContinuousFormulation(Enum):
    VAE = GaussianDistribution
    AE = IdentityDistribution


class DiscreteQuantizer(Enum):
    VQ = VectorQuantizer
    LFQ = LFQuantizer
    FSQ = FSQuantizer
    RESFSQ = ResidualFSQuantizer
