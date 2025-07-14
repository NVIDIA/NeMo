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

from nemo.collections.llm.gpt.model.ssm import HFNemotronHExporter as _NeMo2HFNemotronHExporter
from nemo.collections.llm.gpt.model.ssm import HFNemotronHImporter as _NeMo2HFNemotronHImporter
from nemo.collections.llm.gpt.model.ssm import NemotronHConfigBase
from nemo.tron.converter.common import BaseExporter, BaseImporter
from transformers import AutoConfig as HFAutoConfig

if TYPE_CHECKING:
    #from transformers import Qwen2Config as HFQwen2Config
    ...

logger = logging.getLogger(__name__)


class HFNemotronHExporter(BaseExporter):
    """Exporter to convert NeMo Qwen2 models to Hugging Face format."""

    convert_state = _NeMo2HFNemotronHExporter.convert_state

    @property
    def hf_config(self) -> "HFAutoConfig":
        """Generate a Hugging Face Qwen2 configuration from the NeMo model configuration.

        This property maps NeMo configuration parameters to their Hugging Face equivalents.

        Returns:
            HFQwen2Config: A Hugging Face Qwen2 configuration
        """
        if self._hf_config is not None:
            return self._hf_config

        from transformers import AutoConfig as HFAutoConfig

        source = self.tron_config

        self._hf_config = HFAutoConfig(
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            vocab_size=getattr(source, "vocab_size", self.tokenizer.vocab_size),
            sliding_window=source.seq_length,
            tie_word_embeddings=False,
        )
        return self._hf_config


class HFNemotronHImporter(BaseImporter):
    """Importer for converting Hugging Face Qwen2 models to NeMo Tron format."""

    def init_hf_model(self):
        from transformers import AutoModelForCausalLM

        # Would convert into config (which may be bf16 or fp32)
        #return AutoModelForCausalLM.from_pretrained(str(self.input_path), torch_dtype="auto", trust_remote_code=True)
        # Always converts into fp32
        return AutoModelForCausalLM.from_pretrained(str(self.input_path), trust_remote_code=True)

    convert_state = _NeMo2HFNemotronHImporter.convert_state

    @property
    def hf_config(self) -> "HFAutoConfig":
        from transformers import AutoConfig as HFAutoConfig

        if self._hf_config is not None:
            return self._hf_config
        self._hf_config = HFAutoConfig.from_pretrained(str(self.input_path), trust_remote_code=True)
        return self._hf_config

    @property
    def tron_config(self) -> NemotronHConfigBase:
        """Create a NeMo NemotronHConfig from the HF model config.

        Translates the HF configuration parameters to the equivalent NeMo
        configuration.

        Returns:
            NemotronHConfig: NeMo configuration for NemotronH models
        """
        if self._tron_config is not None:
            return self._tron_config

        self._tron_config = _NeMo2HFNemotronHImporter.config.fget(self.input_path)
        return self._tron_config
