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
from typing import List, Optional
import torch
import torch.nn as nn

from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import instantiate

from nemo.collections.asr.parts.utils import asr_module_utils
from nemo.utils import logging


class AdapterModuleMixin(ABC):
    ADAPTER_CFG: dict

    def add_adapter(self, name: str, cfg: dict):
        if not hasattr(self, 'adapter_layer'):
            self.adapter_layer = nn.ModuleDict()
            AdapterModuleMixin.ADAPTER_CFG = {}

        adapter_enabled = cfg.pop('enabled', True)
        self.adapter_layer[name] = instantiate(cfg)

        cfg['enabled'] = adapter_enabled
        AdapterModuleMixin.ADAPTER_CFG[name] = cfg

    def is_adapter_available(self) -> bool:
        if hasattr(self, 'adapter_layer'):
            return self.adapter_layer is not None and len(self.adapter_layer) > 0
        return False

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        if not self.is_adapter_available():
            raise ValueError("No adapter is available to enable/disable")

        # If name is None, enable/disable all adapters.
        if name is None:
            for name, config in AdapterModuleMixin.ADAPTER_CFG.items():
                AdapterModuleMixin.ADAPTER_CFG[name]['enabled'] = enabled
        else:
            # Enable/Disable just named adapter
            AdapterModuleMixin.ADAPTER_CFG[name]['enabled'] = enabled

    def get_enabled_adapters(self) -> List[str]:
        if not self.is_adapter_available():
            raise ValueError("No adapter is available to get enabled/disabled state")

        enabled_adapters = []
        for name, config in AdapterModuleMixin.ADAPTER_CFG.items():
            if AdapterModuleMixin.ADAPTER_CFG[name]['enabled']:
                enabled_adapters.append(name)

        return enabled_adapters

    def freeze_non_adapter(self) -> None:
        r"""
        Freeze all params for inference.
        """
        for module in self.modules():  # access PT subclass method via inheritance
            for param in module.parameters():
                param.requires_grad = False
            module.eval()

        for module in self.modules():  # access PT subclass method via inheritance
            if hasattr(module, 'adapter_layer') and module.is_adapter_available():
                for name, config in AdapterModuleMixin.ADAPTER_CFG.items():
                    if AdapterModuleMixin.ADAPTER_CFG[name]['enabled']:
                        module.adapter_layer[name].train()

                        for param in module.adapter_layer[name].parameters():
                            param.requires_grad = True

                        logging.info(f"Unfrozen adapter : {name}")



