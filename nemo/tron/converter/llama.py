import logging
from functools import partial
from typing import TYPE_CHECKING

import torch

from nemo.collections.llm.gpt.model.llama import HFLlamaExporter as _NeMo2HFLlamaExporter
from nemo.collections.llm.gpt.model.llama import HFLlamaImporter as _NeMo2HFLlamaImporter
from nemo.collections.llm.gpt.model.llama import (
    Llama31Config,
    LlamaConfig,
)
from nemo.tron.converter.common import (
    BaseImporter,
    dtype_from_hf,
    save_hf_tokenizer_assets,
)
from nemo.tron.tokenizers.tokenizer import _HuggingFaceTokenizer

if TYPE_CHECKING:
    from transformers import LlamaConfig as HFLlamaConfig

logger = logging.getLogger(__name__)


class HFLlamaExporter:
    """Exporter to convert NeMo Llama models to Hugging Face format."""

    convert_state = _NeMo2HFLlamaExporter.convert_state

    @property
    def config(self) -> "HFLlamaConfig":
        """Generate a Hugging Face Llama configuration from the NeMo model configuration.

        This property maps NeMo configuration parameters to their Hugging Face equivalents.

        Returns:
            HFLlamaConfig: A Hugging Face Llama configuration
        """
        if self._config is not None:
            return self._config

        source = self._source_config
        from transformers import LlamaConfig as HFLlamaConfig

        rope_scaling = None
        # For Llama 3.1 and Llama 3.2, rope_scaling is used and thus needed to parsed to the config
        if isinstance(source, Llama31Config):
            rope_scaling = {
                "factor": source.scale_factor,
                "low_freq_factor": source.low_freq_factor,
                "high_freq_factor": source.high_freq_factor,
                "original_max_position_embeddings": source.old_context_len,
                "rope_type": "llama3",
            }

        self._config = HFLlamaConfig(
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            vocab_size=source.vocab_size,
            tie_word_embeddings=source.share_embeddings_and_output_weights,
            rope_scaling=rope_scaling,
            bos_token_id=self.tokenizer.bos_id if self.tokenizer else None,
            eos_token_id=self.tokenizer.eos_id if self.tokenizer else None,
        )
        return self._config


class HFLlamaImporter(BaseImporter):
    """Importer for converting Hugging Face Llama models to NeMo Tron format."""

    def init_hf_model(self):
        from transformers import LlamaForCausalLM

        return LlamaForCausalLM.from_pretrained(str(self.input_path), torch_dtype="auto")

    @property
    def tokenizer(self) -> "_HuggingFaceTokenizer":
        """Get the tokenizer for the HF model.

        Returns:
            _HuggingFaceTokenizer: Tokenizer instance initialized from the HF model's tokenizer
        """

        return _HuggingFaceTokenizer(save_hf_tokenizer_assets(str(self.input_path), str(self.output_path)))

    convert_state = _NeMo2HFLlamaImporter.convert_state

    @property
    def config(self) -> LlamaConfig:
        """Create a NeMo LlamaConfig from the HF model config.

        Translates the HF configuration parameters to the equivalent NeMo
        configuration.

        Returns:
            LlamaConfig: NeMo configuration for Llama models
        """
        from transformers import GenerationConfig
        from transformers import LlamaConfig as HFLlamaConfig

        source = HFLlamaConfig.from_pretrained(str(self.input_path))
        generation_config = GenerationConfig.from_pretrained(str(self.input_path))

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        if getattr(source, "rope_scaling", None) is not None and source.rope_scaling.get("rope_type") == "llama3":
            # Apply Llama3.1 customize rope scaling
            cls = partial(Llama31Config, scale_factor=source.rope_scaling.get("factor", 8.0))
        else:
            cls = LlamaConfig
        output = cls(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
            num_query_groups=source.num_key_value_heads,
            seq_length=source.max_position_embeddings,
            rotary_base=source.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=getattr(source, "tie_word_embeddings", False),
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
            generation_config=generation_config,
            vocab_size=self.tokenizer.vocab_size,
        )

        return output
