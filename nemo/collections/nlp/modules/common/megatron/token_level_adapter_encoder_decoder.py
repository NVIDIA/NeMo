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

from nemo.collections.nlp.modules.common.megatron.token_level_encoder_decoder import (
    MegatronTokenLevelEncoderDecoderModule,
)
from nemo.core import adapter_mixins
from nemo.core.classes.mixins.adapter_mixins import AdapterModuleMixin
from nemo.utils import logging, logging_mode

__all__ = ["MegatronTokenLevelAdapterEncoderDecoderModule"]


class MegatronTokenLevelAdapterEncoderDecoderModule(
    MegatronTokenLevelEncoderDecoderModule, adapter_mixins.AdapterModuleMixin
):
    def add_adapter(self, name: str, cfg: DictConfig):
        self.encoder.add_adapter(name, cfg)
        self.decoder.add_adapter(name, cfg)

    def get_enabled_adapters(self) -> List[str]:
        enabled_adapters = set([])
        for m in [self.encoder, self.decoder]:
            names = m.get_enabled_adapters()
            enabled_adapters.update(names)
        return list(enabled_adapters)

    def set_enabled_adapters(self, name: Optional[str], enabled: bool):
        for m in [self.encoder, self.decoder]:
            m.set_enabled_adapters(name, enabled)

    def is_adapter_available(self) -> bool:
        is_available = any([m.is_adapter_available() for m in [self.encoder, self.decoder]])
        return is_available
