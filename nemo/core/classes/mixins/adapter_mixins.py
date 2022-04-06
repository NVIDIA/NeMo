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

from abc import ABC
from typing import List, Optional

import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from nemo.utils import logging


class AdapterModuleMixin(ABC):
    def add_adapter(self, name: str, cfg: DictConfig):
        if not hasattr(self, 'adapter_layer'):
            self.adapter_layer = nn.ModuleDict()

        if not hasattr(self, 'adapter_cfg'):
            self.adapter_cfg = OmegaConf.create({})

        if name in self.adapter_layer:
            raise ValueError(f"Adapter with name `{name}` already exists !")

        adapter_enabled = cfg.pop('enabled', True)
        self.adapter_layer[name] = instantiate(cfg)

        cfg['enabled'] = adapter_enabled
        self.adapter_cfg[name] = cfg

    def is_adapter_available(self) -> bool:
        if hasattr(self, 'adapter_layer'):
            return self.adapter_layer is not None and len(self.adapter_layer) > 0
        return False

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        if not self.is_adapter_available():
            raise ValueError("No adapter is available to enable/disable")

        # If name is None, enable/disable all adapters.
        if name is None:
            for name, config in self.adapter_cfg.items():
                self.adapter_cfg[name]['enabled'] = enabled
        else:
            # Enable/Disable just named adapter
            self.adapter_cfg[name]['enabled'] = enabled

    def get_enabled_adapters(self) -> List[str]:
        if not self.is_adapter_available():
            raise ValueError("No adapter is available to get enabled/disabled state")

        enabled_adapters = []
        for name, config in self.adapter_cfg.items():
            if self.adapter_cfg[name]['enabled']:
                enabled_adapters.append(name)

        return enabled_adapters

    def unfreeze_enabled_adapters(self, freeze_batchnorm: bool = True) -> None:
        r"""
        Freeze all params for inference.
        """
        if freeze_batchnorm:
            for mname, module in self.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    if hasattr(module, 'weight'):
                        module.weight.requires_grad_(False)
                    if hasattr(module, 'bias'):
                        module.bias.requires_grad_(False)
                    module.eval()
                    module.track_running_stats = False  # prevent running stats from updated during finetuning

                    logging.info(f"Froze module {mname}: {module}")

        adapter_names = set([])
        for module in self.modules():  # access PT subclass method via inheritance
            if hasattr(module, 'adapter_layer') and module.is_adapter_available():
                for name, config in self.adapter_cfg.items():
                    if self.adapter_cfg[name]['enabled']:
                        module.adapter_layer[name].train()

                        for pname, param in module.adapter_layer[name].named_parameters():
                            param.requires_grad = True

                        # unfreeze batch norm if any
                        for mname, module in module.adapter_layer[name].named_modules():
                            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                                module.track_running_stats = (
                                    True  # prevent running stats from updated during finetuning
                                )
                                logging.info(f"Unfroze adapter module {mname}: {module}")

                        adapter_names.add(name)

        for name in adapter_names:
            logging.info(f"Unfrozen adapter : {name}")
