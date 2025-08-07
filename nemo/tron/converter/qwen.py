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
from functools import partial
from typing import TYPE_CHECKING

from nemo.collections.llm.gpt.model.qwen2 import HFQwen2Exporter as _NeMo2HFQwen2Exporter
from nemo.collections.llm.gpt.model.qwen2 import HFQwen2Importer as _NeMo2HFQwen2Importer
from nemo.collections.llm.gpt.model.qwen2 import Qwen2Config

from nemo.collections.llm.gpt.model.qwen3 import HFQwen3Exporter as _NeMo2HFQwen3Exporter
from nemo.collections.llm.gpt.model.qwen3 import HFQwen3Importer as _NeMo2HFQwen3Importer
from nemo.collections.llm.gpt.model.qwen3 import Qwen3Config

from nemo.tron.converter.common import BaseExporter, BaseImporter

if TYPE_CHECKING:
    from transformers import Qwen2Config as HFQwen2Config
    from transformers import Qwen3Config as HFQwen3Config
    from transformers import Qwen3MoeConfig as HFQwen3MoeConfig

logger = logging.getLogger(__name__)


class HFQwen2Exporter(BaseExporter):
    """Exporter to convert NeMo Qwen2 models to Hugging Face format."""

    convert_state = _NeMo2HFQwen2Exporter.convert_state

    @property
    def hf_config(self) -> "HFQwen2Config":
        """Generate a Hugging Face Qwen2 configuration from the NeMo model configuration.

        This property maps NeMo configuration parameters to their Hugging Face equivalents.

        Returns:
            HFQwen2Config: A Hugging Face Qwen2 configuration
        """
        if self._hf_config is not None:
            return self._hf_config

        from transformers import Qwen2Config as HFQwen2Config

        source = self.tron_config

        self._hf_config = HFQwen2Config(
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
            tie_word_embeddings=source.share_embeddings_and_output_weights,
        )
        return self._hf_config


class HFQwen2Importer(BaseImporter):
    """Importer for converting Hugging Face Qwen2 models to NeMo Tron format."""

    def init_hf_model(self):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(str(self.input_path), torch_dtype="auto", trust_remote_code=True)

    convert_state = _NeMo2HFQwen2Importer.convert_state

    @property
    def hf_config(self) -> "HFQwen2Config":
        from transformers import Qwen2Config as HFQwen2Config

        if self._hf_config is not None:
            return self._hf_config
        self._hf_config = HFQwen2Config.from_pretrained(str(self.input_path), trust_remote_code=True)
        return self._hf_config

    @property
    def tron_config(self) -> Qwen2Config:
        """Create a NeMo Qwen2Config from the HF model config.

        Translates the HF configuration parameters to the equivalent NeMo
        configuration.

        Returns:
            Qwen2Config: NeMo configuration for Qwen2 models
        """
        if self._tron_config is not None:
            return self._tron_config

        self._tron_config = _NeMo2HFQwen2Importer.config.fget(self.input_path)
        return self._tron_config


class HFQwen3Exporter(BaseExporter):
    """Exporter to convert NeMo Qwen3 models to Hugging Face format."""

    convert_state = _NeMo2HFQwen3Exporter.convert_state

    @property
    def hf_config(self) -> "HFQwen3Config":
        """Generate a Hugging Face Qwen3 configuration from the NeMo model configuration.

        This property maps NeMo configuration parameters to their Hugging Face equivalents.

        Returns:
            HFQwen3Config: A Hugging Face Qwen3 configuration
        """
        if self._hf_config is not None:
            return self._hf_config

        from transformers import Qwen3Config as HFQwen3Config
        from transformers import Qwen3MoeConfig as HFQwen3MoeConfig

        source = self.tron_config
        
        is_moe = source.num_moe_experts is not None
        hf_config_cls = (
            partial(
                HFQwen3MoeConfig,
                moe_intermediate_size=source.moe_ffn_hidden_size,
                num_experts=source.num_moe_experts,
                num_experts_per_tok=source.moe_router_topk,
                router_aux_loss_coef=source.moe_aux_loss_coeff,
                norm_topk_prob=True,
            )
            if is_moe
            else HFQwen3Config
        )

        self._hf_config = hf_config_cls(
            architectures=["Qwen3ForCausalLM"],
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            head_dim=source.kv_channels,
            max_position_embeddings=source.max_position_embeddings,
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            vocab_size=getattr(source, "vocab_size",
                               self.tokenizer.vocab_size),
            sliding_window=None,
            tie_word_embeddings=source.share_embeddings_and_output_weights,
            max_window_layers=source.num_layers,
            bos_token_id=151643,
            eos_token_id=151645,
        )
        return self._hf_config


class HFQwen3Importer(BaseImporter):
    """Importer for converting Hugging Face Qwen3 models to NeMo Tron format."""

    def init_hf_model(self):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(str(self.input_path), torch_dtype="auto", trust_remote_code=True)

    convert_state = _NeMo2HFQwen3Importer.convert_state

    @property
    def hf_config(self) -> "HFQwen3Config":
        from transformers import Qwen3Config as HFQwen3Config

        if self._hf_config is not None:
            return self._hf_config
        self._hf_config = HFQwen3Config.from_pretrained(str(self.input_path), trust_remote_code=True)
        return self._hf_config

    @property
    def tron_config(self) -> Qwen3Config:
        """Create a NeMo Qwen3Config from the HF model config.

        Translates the HF configuration parameters to the equivalent NeMo
        configuration.

        Returns:
            Qwen3Config: NeMo configuration for Qwen3 models
        """
        if self._tron_config is not None:
            return self._tron_config

        self._tron_config = _NeMo2HFQwen3Importer.config.fget(self.input_path)
        return self._tron_config

    @property
    def config(self) -> Qwen3Config:
        return self.tron_config
