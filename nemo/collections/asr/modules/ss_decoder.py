# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf

from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    LengthsType,
    LogitsType,
    LogprobsType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging

__all__ = ['ConvASRDecoder', 'ConvASREncoder', 'ConvASRDecoderClassification']


class ConvSSDecoder(NeuralModule, Exportable):
    """
    Decoder layer to reconstruct speech from encoded

    Args:
        kernel_size (int): 
        in_channels (int): 
        out_channels (int):
        stride (int):
        bias (bool): 
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True):
        super().__init__()
        self.decoder = nn.ConvTranspose1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=bias,
        )

    def forward(self, x):
        """
        Args:
            x : torch.Tensor
                [B, F, L]
        """
        if not x.dim() == 3:
            print(f"x shape in decoder is {x.shape}")
            raise RuntimeError(f"Expecting a 3-dim tensor but got {x.dim()} dims")

        x = self.decoder(x)
        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)

        return x
