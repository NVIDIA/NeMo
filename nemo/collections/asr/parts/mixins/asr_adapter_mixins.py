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

from typing import List, Optional

from omegaconf import DictConfig, OmegaConf, open_dict
from dataclasses import is_dataclass

from nemo.core.classes.mixins.adapter_mixins import AdapterModuleMixin
from nemo.utils import logging


class ASREncoderAdapterModelMixin(AdapterModuleMixin):
    """ ASR Adapter Mixin that can augment any Encoder module with Adapter module support.

    This mixin class should be used only with a top level ModelPT subclass, that includes an `encoder` submodule.
    This mixin class adds several utility methods which are propagated to the `encoder`.

    An Adapter module is any Pytorch nn.Module that possess a few properties :

    - It's input and output dimension are the same, while the hidden dimension need not be the same.
    - The final layer of the Adapter module is zero-initialized, so that the residual connection to the adapter
        yields the original output.

    This mixin adds the following instance variables to the class this inherits it:

        -   `adapter_layer`: A torch.nn.ModuleDict(), whose keys are the names of the adapter (globally unique),
                and values are the Adapter nn.Module().
        -   `adapter_cfg`: A OmegaConf DictConfig object that holds the config of the adapters that are initialized.

    **Note**: This module **is** responsible for maintaining its config. At the ModelPT level, it will access and
        write Adapter config information to `self.cfg.adapters`.
    """

    def setup_encoder_adapters(self):
        """
        Utility method that is called in the ASR ModelPT-implementation constructor, so as to restore any
        adapters that were previously added.

        This method should be called just once at constructor time.
        """
        if not hasattr(self, 'encoder') or not isinstance(self.encoder, AdapterModuleMixin):
            return

        # Test if `adapters` is part of the config (injected from previous Adapter additions)
        if 'adapters' in self.cfg:
            # Set the global config of adapters
            self._update_adapter_cfg(self.cfg.adapters)

            # Dispatch the call to the encoder, for every adapter contained in the config.
            for adapter_name, adapter_cfg in self.cfg.adapters.items():
                self.add_adapter(name=adapter_name, cfg=adapter_cfg)
                logging.info(
                    f"Finished setup of adapter : '{adapter_name}'. Enabled: {adapter_cfg.get('enabled', True)}."
                )

    def add_adapter(self, name: str, cfg: DictConfig):
        """
        Add an Adapter module to this model.

        Args:
            name: A globally unique name for the adapter. Will be used to access, enable and disable adapters.
            cfg: A DictConfig that contains at the bare minimum `__target__` to instantiate a new Adapter module.
        """
        self._check_valid_model_with_adapter_support()

        # Convert to DictConfig from dict or Dataclass
        if is_dataclass(cfg):
            cfg = OmegaConf.structured(cfg)

        if not isinstance(cfg, DictConfig):
            cfg = DictConfig(cfg)

        # Update the model.cfg with information about the new adapter from cfg
        with open_dict(cfg), open_dict(self.cfg):
            if 'adapters' not in self.cfg:
                self.cfg.adapters = OmegaConf.create({})

            if 'enabled' not in cfg:
                cfg['enabled'] = True

            self.cfg.adapters[name] = OmegaConf.create(cfg)

            # Set the global config of adapters
            self._update_adapter_cfg(self.cfg.adapters)

            # Dispatch the call to the encoder.
            self.encoder.add_adapter(name=name, cfg=self.cfg.adapters[name])

    def is_adapter_available(self) -> bool:
        """
        Checks if any Adapter module has been instantiated.

        Returns:
            bool, determining if any Adapter module has been instantiated. Returns true even if the adapters are
            enabled or disabled, false only if no adapters exist.
        """
        self._check_valid_model_with_adapter_support()
        return self.encoder.is_adapter_available()

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        """
        Updated the internal adapter config, determining if an adapter (or all adapters) are either
        enabled or disabled.

        A common user pattern would be to disable all adapters (either after adding them, or restoring a model
        with pre-existing adapters) and then simply enable one of the adapters.

        .. code::

            model.set_enabled_adapters(enabled=False)
            model.set_enabled_adapters(name=<some adapter name>, enabled=True)

        Args:
            name: Optional str. If a str name is given, the config will be updated to the value of `enabled`.
                If no name is given, then all adapters will be enabled/disabled.
            enabled: Bool, determines if the adapter(s) will be enabled/disabled.
        """
        self._check_valid_model_with_adapter_support()

        # Update the adapter config with information about whether it is enabled/disabled.
        with open_dict(self.cfg.adapters):
            # If no name is provided, update all adapters.
            if name is None:
                for key in self.cfg.adapters.keys():
                    self.cfg.adapters[key]['enabled'] = enabled
                    logging.info(f"Setting adapter '{key}' status : Enabled = {enabled}")

            else:
                # Otherwise, update just the specified adapter.
                self.cfg.adapters[name]['enabled'] = enabled
                logging.info(f"Setting adapter '{name}' status : Enabled = {enabled}")

            # Dispatch the call to the encoder.
            self.encoder.set_enabled_adapters(name=name, enabled=enabled)

    def get_enabled_adapters(self) -> List[str]:
        """
        Returns a list of all enabled adapters.

        Returns:
            A list of str names of each enabled adapter(s).
        """
        self._check_valid_model_with_adapter_support()

        enabled_adapters = self.encoder.get_enabled_adapters()
        return enabled_adapters

    def _check_valid_model_with_adapter_support(self):
        """
        Utility method to test if the subclass of this mixin is an appropriate subclass of ModelPT itself.
        """
        if not hasattr(self, 'encoder'):
            raise ValueError("Cannot add adapter to this object as it does not have an `encoder` sub-module!")

        if not isinstance(self.encoder, AdapterModuleMixin):
            raise ValueError(f'{self.encoder.__class__.__name__} does not implement `AdapterModuleMixin`')

    def _update_adapter_cfg(self, cfg: DictConfig):
        """
        Utility method to recursively update all of the Adapter module configs with the provided config.
        **Note**: It is not a (deep)copy, but a reference copy. Changes made to the config will be reflected to
            adapter submodules, but it is still encouraged to explicitly update the adapter_cfg using this method.

        Args:
            cfg: DictConfig containing the value of `model.cfg.adapters`.
        """
        for module in self.modules():  # access PT subclass method via inheritance
            if isinstance(module, AdapterModuleMixin):
                module.adapter_cfg = cfg
