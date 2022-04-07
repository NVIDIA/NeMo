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
from dataclasses import is_dataclass
from typing import List, Optional

import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.utils import logging


class AdapterModuleMixin(ABC):
    """ Generic Adapter Mixin that can augment any torch.nn.Module with Adapter module support.

    This mixin class adds a hierarchical way to add any type of Adapter modules to a pre-existing module.
    Since Models are inherently also nn.Module, this mixin can be attached to any Model or Module.
    This mixin class adds several utility methods which are utilized or overridden as necessary.

    An Adapter module is any Pytorch nn.Module that possess a few properties :

        -   It's input and output dimension are the same, while the hidden dimension need not be the same.
        -   The final layer of the Adapter module is zero-initialized, so that the residual connection to the adapter
                yields the original output.

    This mixin adds the following instance variables to the class this inherits it:

        -   `adapter_layer`: A torch.nn.ModuleDict(), whose keys are the names of the adapter (globally unique),
                and values are the Adapter nn.Module().
        -   `adapter_cfg`: A OmegaConf DictConfig object that holds the config of the adapters that are initialized.

    **Note**: This module is **not** responsible for maintaining its config. Subclasses must ensure config is updated
        or preserved as needed. It is the responsibility of the subclasses to propagate the most up to date config to
        lower layers.
    """

    def add_adapter(self, name: str, cfg: DictConfig):
        """
        Add an Adapter module to this module.

        Args:
            name: A globally unique name for the adapter. Will be used to access, enable and disable adapters.
            cfg: A DictConfig or Dataclass that contains at the bare minimum `__target__` to instantiate a
                new Adapter module.
        """
        # Convert to DictConfig from dict or Dataclass
        if is_dataclass(cfg):
            cfg = OmegaConf.structured(cfg)

        if not isinstance(cfg, DictConfig):
            cfg = DictConfig(cfg)

        # Add adapter_layer ModuleDict() if not present.
        if not hasattr(self, 'adapter_layer'):
            self.adapter_layer = nn.ModuleDict()

        # Add adapter_cfg if it doesnt exist or hasnt been assigned yet.
        if not hasattr(self, 'adapter_cfg'):
            self.adapter_cfg = OmegaConf.create({})

        # Assert that name is globally unique to all adapters.
        if name in self.adapter_layer:
            raise ValueError(f"Adapter with name `{name}` already exists !")

        # Update internal config and instantiate the Adapter module
        with open_dict(cfg), open_dict(self.adapter_cfg):
            adapter_enabled = cfg.pop('enabled', True)
            self.adapter_layer[name] = instantiate(cfg)

            cfg['enabled'] = adapter_enabled
            self.adapter_cfg[name] = cfg

    def is_adapter_available(self) -> bool:
        """
        Checks if any Adapter module has been instantiated.

        Returns:
            bool, determining if any Adapter module has been instantiated. Returns true even if the adapters are
            enabled or disabled, false only if no adapters exist.
        """
        if hasattr(self, 'adapter_layer'):
            return self.adapter_layer is not None and len(self.adapter_layer) > 0
        return False

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        """
        Updated the internal adapter config, determining if an adapter (or all adapters) are either
        enabled or disabled.

        A common user pattern would be to disable all adapters (either after adding them, or restoring a model
        with pre-existing adapters) and then simply enable one of the adapters.

        .. code::

            module.set_enabled_adapters(enabled=False)
            module.set_enabled_adapters(name=<some adapter name>, enabled=True)

        Args:
            name: Optional str. If a str name is given, the config will be updated to the value of `enabled`.
                If no name is given, then all adapters will be enabled/disabled.
            enabled: Bool, determines if the adapter(s) will be enabled/disabled.
        """
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
        """
        Returns a list of all enabled adapters.

        Returns:
            A list of str names of each enabled adapter(s).
        """
        if not self.is_adapter_available():
            raise ValueError("No adapter is available to get enabled/disabled state")

        enabled_adapters = []
        for name, config in self.adapter_cfg.items():
            if self.adapter_cfg[name]['enabled']:
                enabled_adapters.append(name)

        return enabled_adapters

    def unfreeze_enabled_adapters(self, freeze_batchnorm: bool = True) -> None:
        """
        Utility method to unfreeze only the enabled Adapter module(s).

        A common user pattern is to freeze all the modules (including all the adapters), and then
        unfreeze just the required adapters.

        .. code::

            module.freeze()  # only available to nemo.core.NeuralModule !
            module.unfreeze_enabled_adapters()

        Args:
            freeze_batchnorm: An optional (and recommended) practice of freezing the updates to the moving average
                buffers of any and all BatchNorm*D layers. This is necessary to ensure that disabling all adapters
                will precisely yield the original (base) model's outputs.
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
