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

import os
from abc import ABC, abstractmethod
from typing import List
import torch
import torch.nn as nn

from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.parts.utils import asr_module_utils
from nemo.utils import logging


class AdapterModuleMixin(ABC):

    def add_adapter(self, dim: int):
        self.adapter_layer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1000, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(1000, dim, bias=False)
        )
        self.adapter_layer[-1].weight.data *= 0

    def is_adapter_available(self) -> bool:
        if hasattr(self, 'adapter_layer'):
            return self.adapter_layer is not None
        return False

    def freeze_non_adapter(self) -> None:
        r"""
        Freeze all params for inference.
        """
        for param in self.parameters():
            param.requires_grad = False

        for param in self.adapter_layer.parameters():
            param.requires_grad = True

        self.eval()
        self.adapter_layer.train()



