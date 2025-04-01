import logging
from typing import TYPE_CHECKING

from nemo.collections.llm.gpt.model.qwen2 import HFQwen2Exporter as _NeMo2HFQwen2Exporter
from nemo.collections.llm.gpt.model.qwen2 import HFQwen2Importer as _NeMo2HFQwen2Importer
from nemo.collections.llm.gpt.model.qwen2 import Qwen2Config
from nemo.tron.converter.common import BaseExporter, BaseImporter

if TYPE_CHECKING:
    from transformers import Qwen2Config as HFQwen2Config

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
            tie_word_embeddings=False,
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
