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

"""Transformer based language model with Adapter support."""

from nemo.collections.nlp.modules.common.megatron.megatron_transformer_encoder import MegatronTransformerEncoderModule
from nemo.core import adapter_mixins
from typing import List, Optional
from omegaconf import DictConfig


__all__ = ["MegatronTransformerAdapterEncoderModule"]


class MegatronTransformerAdapterEncoderModule(MegatronTransformerEncoderModule, adapter_mixins.AdapterModuleMixin):
    """Transformer decoder model.
    """
    def add_adapter(self, name: str, cfg: DictConfig):
      self.model.add_adapter(name, cfg)
      
    def get_enabled_adapters(self) -> List[str]:
        return list(self.model.get_enabled_adapters())
    
    def set_enabled_adapters(self, name: Optional[str], enabled: bool):
        self.model.set_enabled_adapters(name, enabled)
    
    def is_adapter_available(self) -> bool:
        return self.model.is_adapter_available()
