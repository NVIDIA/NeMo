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
from nemo.core import ModelPT
from nemo.utils import logging


class AdapterModuleMixin(ABC):

    def add_adapter(self, name: str, cfg: DictConfig):
        if not hasattr(self, 'adapter_layer'):
            self.adapter_layer = nn.ModuleDict()

        if not hasattr(self, 'ADAPTER_CFG') or self.ADAPTER_CFG is None:
            self.ADAPTER_CFG = OmegaConf.create({})

        if name in self.adapter_layer:
            raise ValueError(f"Adapter with name `{name}` already exists !")

        adapter_enabled = cfg.pop('enabled', True)
        self.adapter_layer[name] = instantiate(cfg)

        cfg['enabled'] = adapter_enabled
        self.ADAPTER_CFG[name] = cfg

    def is_adapter_available(self) -> bool:
        if hasattr(self, 'adapter_layer'):
            return self.adapter_layer is not None and len(self.adapter_layer) > 0
        return False

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        if not self.is_adapter_available():
            raise ValueError("No adapter is available to enable/disable")

        # If name is None, enable/disable all adapters.
        if name is None:
            for name, config in self.ADAPTER_CFG.items():
                self.ADAPTER_CFG[name]['enabled'] = enabled
        else:
            # Enable/Disable just named adapter
            self.ADAPTER_CFG[name]['enabled'] = enabled

    def get_enabled_adapters(self) -> List[str]:
        if not self.is_adapter_available():
            raise ValueError("No adapter is available to get enabled/disabled state")

        enabled_adapters = []
        for name, config in self.ADAPTER_CFG.items():
            if self.ADAPTER_CFG[name]['enabled']:
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
                for name, config in self.ADAPTER_CFG.items():
                    if self.ADAPTER_CFG[name]['enabled']:
                        module.adapter_layer[name].train()

                        for pname, param in module.adapter_layer[name].named_parameters():
                            param.requires_grad = True

                        # unfreeze batch norm if any
                        for mname, module in module.adapter_layer[name].named_modules():
                            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                                module.track_running_stats = True  # prevent running stats from updated during finetuning
                                logging.info(f"Unfroze adapter module {mname}: {module}")

                        adapter_names.add(name)

        for name in adapter_names:
            logging.info(f"Unfrozen adapter : {name}")


class EncoderAdapterModelMixin(AdapterModuleMixin):

    def setup_encoder_adapters(self):
        if not isinstance(self, ModelPT) or not isinstance(self.encoder, AdapterModuleMixin):
            return

        if 'adapters' in self.cfg:
            # Set the global config of adapters
            self._update_adapter_cfg(self.cfg.adapters)

            for adapter_name, adapter_cfg in self.cfg.adapters.items():
                self.add_adapter(name=adapter_name, cfg=adapter_cfg)

    def add_adapter(self, name: str, cfg: DictConfig):
        self._check_valid_model_with_adapter_support()

        with open_dict(self.cfg):
            # if 'adapters' in self.cfg and name in self.cfg.adapters:
            #     raise ValueError(f"Adapter with name {name} already exists in this model !")

            if 'adapters' not in self.cfg:
                self.cfg.adapters = OmegaConf.create({})

            if 'enabled' not in cfg:
                cfg['enabled'] = True

            self.cfg.adapters[name] = OmegaConf.create(cfg)

            # Set the global config of adapters
            self._update_adapter_cfg(self.cfg.adapters)

            self.encoder.add_adapter(name=name, cfg=self.cfg.adapters[name])

    def is_adapter_available(self) -> bool:
        self._check_valid_model_with_adapter_support()
        return self.encoder.is_adapter_available()

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        self._check_valid_model_with_adapter_support()

        with open_dict(self.cfg.adapters):
            if name is None:
                for key in self.cfg.adapters.keys():
                    self.cfg.adapters[key]['enabled'] = enabled

            else:
                self.cfg.adapters[name]['enabled'] = enabled

            self.encoder.set_enabled_adapters(name=name, enabled=enabled)

    def get_enabled_adapters(self) -> List[str]:
        self._check_valid_model_with_adapter_support()

        enabled_adapters = self.encoder.get_enabled_adapters()
        return enabled_adapters

    def _check_valid_model_with_adapter_support(self):
        if not isinstance(self, ModelPT):
            raise ValueError("Cannot add adapter to this object as it does not inherit ModelPT!")

        if not isinstance(self.encoder, AdapterModuleMixin):
            raise ValueError(f'{self.encoder.__class__.__name__} does not implement `AdapterModuleMixin`')

    def _update_adapter_cfg(self, cfg: DictConfig):
        for module in self.modules():  # access PT subclass method via inheritance
            if isinstance(module, AdapterModuleMixin):
                module.ADAPTER_CFG = cfg
