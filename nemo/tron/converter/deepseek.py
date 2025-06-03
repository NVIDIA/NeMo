# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from typing import TYPE_CHECKING

from nemo.collections.llm.gpt.model.deepseek import (
    HFDeepSeekImporter as _NeMo2HFDeepSeekImporter,
)
from nemo.collections.llm.gpt.model.deepseek import DeepSeekConfig
from nemo.tron.converter.common import BaseImporter

if TYPE_CHECKING:
    from transformers import DeepSeekConfig as HFDeepSeekConfig

logger = logging.getLogger(__name__)



class HFDeepSeekImporter(BaseImporter):
    """Importer for converting Hugging Face DeepSeek models to NeMo Tron format."""
    def init_hf_model(self):
        from transformers import AutoModelForCausalLM

        self._importer = _NeMo2HFDeepSeekImporter(self.input_path)
        self._importer.convert_mtp = False # MTP not supported yet

        return AutoModelForCausalLM.from_pretrained(
            str(self.input_path), torch_dtype="auto", trust_remote_code=True
        )

    def convert_state(self, source, target):
        from megatron.core.transformer.module import MegatronModule

        class ModuleWrapper:
            """Simple wrapper class that provides access to an object via .module attribute."""
            def __init__(self, target):
                self.module = target

        # `convert_state` expects target to be wrapped in a .module
        wrapped_target = ModuleWrapper(target)

        # `apply_transforms` expects target to have a `named_parameters` function
        wrapped_target.named_parameters = target.named_parameters

        wrapped_target = self._importer.convert_state(source, wrapped_target)
        return wrapped_target.module

    @property
    def hf_config(self) -> "HFDeepSeekConfig":
        # from transformers import DeepSeekV3Config as HFDeepSeekConfig
        from transformers import AutoConfig as HFDeepSeekConfig

        if self._hf_config is not None:
            return self._hf_config
        self._hf_config = HFDeepSeekConfig.from_pretrained(
            str(self.input_path), trust_remote_code=True
        )
        return self._hf_config

    @property
    def tron_config(self) -> DeepSeekConfig:
        """Create a NeMo DeepSeekConfig from the HF model config.

        Translates the HF configuration parameters to the equivalent NeMo
        configuration.

        Returns:
            DeepSeekConfig: NeMo configuration for DeepSeek models
        """
        if self._tron_config is not None:
            return self._tron_config

        self._tron_config = self._importer.config
        return self._tron_config

    @property
    def config(self) -> DeepSeekConfig:
        return self.tron_config
