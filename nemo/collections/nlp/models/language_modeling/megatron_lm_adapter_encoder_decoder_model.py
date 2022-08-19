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

from typing import List, Optional

from omegaconf import DictConfig

from nemo.core import adapter_mixins
from nemo.core.classes.mixins.adapter_mixins import AdapterModuleMixin
from nemo.utils import logging, logging_mode

__all__ = ["MegatronLMAdapterEncoderDecoderModel"]


class MegatronLMAdapterEncoderDecoderModel(adapter_mixins.AdapterModelPTMixin):
    """ encoder-decoder T5 adapter model """

    # override functions
    def setup_adapters(self):
        # First check that any of the modules support adapters or not
        supports_adapters = False

        # Check the inheriting class' modules supports adapters or not
        if hasattr(self, 'encoder') and isinstance(self.encoder, AdapterModuleMixin):
            supports_adapters |= True

        if hasattr(self, 'decoder') and isinstance(self.decoder, AdapterModuleMixin):
            supports_adapters |= True

        # If any class supports it, try to restore adapters
        if supports_adapters:
            super().setup_adapters()

    def add_adapter(self, name: str, cfg: DictConfig):
        # Setup the config first
        super().add_adapter(name, cfg)

        # Resolve module name and adapter name
        module_name, adapter_name = self.resolve_adapter_module_name_(name)

        # Try to retrieve global adapter config
        global_config = self._get_global_cfg()

        # forward the method call to the individual modules
        # If module name is empty, it is a default and global adapter, otherwise it is a module adapter
        if (module_name == '' and global_config.get('encoder_adapter', True)) or (module_name == 'encoder'):
            self.encoder.add_adapter(name, cfg)

        if (module_name == '' and global_config.get('decoder_adapter', False)) or (module_name == 'decoder'):
            self.decoder.add_adapter(name, cfg)

    def resolve_adapter_module_name_(self, name: str):
        # resolve name and module
        module_name, adapter_name = super().resolve_adapter_module_name_(name)

        # '' as module name means "default module"
        # assert that the module name (if provided) is valid - default, encoder or decoder
        valid_module_names = self.adapter_module_names  # Get the list of supported adapter modules from property
        if module_name not in valid_module_names:
            raise ValueError(f"Provided module name `{module_name}` is not in valid list : {valid_module_names}")

        return (module_name, adapter_name)

    def _get_global_cfg(self):
        # Utility method to get a default "global" adapter config (can be given any value by the user in this config)
        global_config = DictConfig({})
        if 'adapters' in self.cfg and self.adapter_global_cfg_key in self.cfg.adapters:
            global_config = self.adapter_cfg[self.adapter_global_cfg_key]
        return global_config

    @property
    def adapter_module_names(self) -> List[str]:
        module_names = super().adapter_module_names  # "Default" adapter module: ''
        module_names.extend(['encoder', 'decoder'])  # Add support for `encoder` and `decoder` modules
        return module_names

    def get_enabled_adapters(self) -> List[str]:
        enabled_adapters = super().get_enabled_adapters()

        # Forward the method call to the individual modules
        if isinstance(self.encoder, AdapterModuleMixin):
            encoder_adapters = self.encoder.get_enabled_adapters()
            enabled_adapters.extend(encoder_adapters)

        if isinstance(self.decoder, AdapterModuleMixin):
            decoder_adapters = self.decoder.get_enabled_adapters()
            enabled_adapters.extend(decoder_adapters)

        return enabled_adapters

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        # check if valid model with some adapter support
        super().set_enabled_adapters(name, enabled)

        # Resolve module name and adapter name
        if name is not None:
            module_name, _ = self.resolve_adapter_module_name_(name)
        else:
            module_name = None

        # Try to retrieve global adapter config
        global_config = self._get_global_cfg()

        if name is None or global_config.get('encoder_adapter', True) or module_name in ('', 'encoder'):
            if self.encoder.is_adapter_available():
                self.encoder.set_enabled_adapters(name, enabled)

        if name is None or global_config.get('decoder_adapter', False) or module_name == 'decoder':
            if self.decoder.is_adapter_available():
                self.decoder.set_enabled_adapters(name, enabled)

    def check_valid_model_with_adapter_support_(self):
        global_cfg = DictConfig({})
        if self.adapter_global_cfg_key in self.adapter_cfg:
            global_cfg = self.adapter_cfg[self.adapter_global_cfg_key]

        encoder_adapter = global_cfg.get('encoder_adapter', True)
        decoder_adapter = global_cfg.get('decoder_adapter', False)

        if encoder_adapter and not hasattr(self, 'encoder'):
            logging.warning("Encoder not available", mode=logging_mode.ONCE)
        elif encoder_adapter and not isinstance(self.encoder, AdapterModuleMixin):
            logging.warning("Encoder does not support adapters !", mode=logging_mode.ONCE)

        if decoder_adapter and not hasattr(self, 'decoder'):
            logging.warning("Decoder is not available", mode=logging_mode.ONCE)
        elif decoder_adapter and not isinstance(self.decoder, AdapterModuleMixin):
            logging.warning("Decoder does not support adapters !", mode=logging_mode.ONCE)
