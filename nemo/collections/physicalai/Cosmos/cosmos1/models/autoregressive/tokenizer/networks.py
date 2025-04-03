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

from collections import namedtuple

import torch
from torch import nn

from cosmos1.models.autoregressive.tokenizer.modules import CausalConv3d, DecoderFactorized, EncoderFactorized
from cosmos1.models.autoregressive.tokenizer.quantizers import FSQuantizer
from cosmos1.utils import log

NetworkEval = namedtuple("NetworkEval", ["reconstructions", "quant_loss", "quant_info"])


class CausalDiscreteVideoTokenizer(nn.Module):
    def __init__(self, z_channels: int, z_factor: int, embedding_dim: int, **kwargs) -> None:
        super().__init__()
        self.name = kwargs.get("name", "CausalDiscreteVideoTokenizer")
        self.embedding_dim = embedding_dim
        self.encoder = EncoderFactorized(z_channels=z_factor * z_channels, **kwargs)
        self.decoder = DecoderFactorized(z_channels=z_channels, **kwargs)

        self.quant_conv = CausalConv3d(z_factor * z_channels, embedding_dim, kernel_size=1, padding=0)
        self.post_quant_conv = CausalConv3d(embedding_dim, z_channels, kernel_size=1, padding=0)

        self.quantizer = FSQuantizer(**kwargs)

        num_parameters = sum(param.numel() for param in self.parameters())
        log.debug(f"model={self.name}, num_parameters={num_parameters:,}")
        log.debug(f"z_channels={z_channels}, embedding_dim={self.embedding_dim}.")

    def to(self, *args, **kwargs):
        setattr(self.quantizer, "dtype", kwargs.get("dtype", torch.bfloat16))
        return super(CausalDiscreteVideoTokenizer, self).to(*args, **kwargs)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return self.quantizer(h)

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        return self.decoder(quant)

    def forward(self, input):
        quant_info, quant_codes, quant_loss = self.encode(input)
        reconstructions = self.decode(quant_codes)
        if self.training:
            return dict(reconstructions=reconstructions, quant_loss=quant_loss, quant_info=quant_info)
        return NetworkEval(reconstructions=reconstructions, quant_loss=quant_loss, quant_info=quant_info)
