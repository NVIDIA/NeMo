# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from torch import nn as nn

from nemo.collections.asr.parts.submodules.jasper import jasper_activations


class LinearAdapter(nn.Module):
    def __init__(self, in_features, dim, activation: str = 'swish'):
        super().__init__()

        activation = jasper_activations[activation]()
        # If the activation can be executed in place, do so.
        if hasattr(activation, 'inplace'):
            activation.inplace = True

        self.module = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, dim, bias=False),
            activation,
            nn.Linear(dim, in_features, bias=False),
        )
        self.module[-1].weight.data *= 0

    def forward(self, x):
        return self.module(x)
