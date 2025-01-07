from dataclasses import dataclass
from typing import TYPE_CHECKING, Union, Callable, Annotated, Optional

import nemo.collections.llm.gpt.model.base as GPTBase
from nemo.collections.llm.gpt.model import GPTConfig
from nemo.collections.llm.gpt.model.llama import Llama32Config1B, LlamaConfig, HFLlamaImporter, LlamaModel
from nemo.utils.import_utils import safe_import
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.enums import AttnMaskType
from nemo.lightning import OptimizerModule, io, teardown
from nemo.collections.llm.utils import Config
from nemo.lightning.pytorch.utils import dtype_from_hf
import torch
from torch import nn
if TYPE_CHECKING:
    from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
    from transformers import LlamaConfig as HFLlamaConfig
    from transformers import LlamaForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
_, HAVE_TE = safe_import("transformer_engine")

def local_layer_spec(config: "GPTConfig") -> ModuleSpec:
    gpt_layer_spec = GPTBase.local_layer_spec(config)
    gpt_layer_spec.submodules.self_attention.params['attn_mask_type'] = AttnMaskType.padding
    return gpt_layer_spec

def transformer_engine_layer_spec(config: "GPTConfig") -> ModuleSpec:
    gpt_layer_spec = GPTBase.transformer_engine_layer_spec(config)
    gpt_layer_spec.submodules.self_attention.params['attn_mask_type'] = AttnMaskType.padding
    return gpt_layer_spec

def get_nv_embedding_layer_spec(config):
    if HAVE_TE:
        return local_layer_spec(config)
        return transformer_engine_layer_spec(config)
    else:
        return local_layer_spec(config)

@dataclass
class NVEmbedLlama32Config1B(Llama32Config1B):
    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTConfig"], ModuleSpec]] = get_nv_embedding_layer_spec

class NVEmbedLlamaModel(LlamaModel):
    def __init__(
        self,
        config: Annotated[Optional[LlamaConfig], Config[LlamaConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config or LlamaConfig(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)


@io.model_importer(NVEmbedLlamaModel, "hf")
class HFNVEmbedLlamaImporter(HFLlamaImporter):
    def init(self) -> NVEmbedLlamaModel:
        return NVEmbedLlamaModel(self.config, tokenizer=self.tokenizer)

    @property
    def config(self) -> Llama32Config1B:
        from transformers import LlamaConfig as HFLlamaConfig

        source = HFLlamaConfig.from_pretrained(str(self))

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        output = NVEmbedLlama32Config1B(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
            num_query_groups=source.num_key_value_heads,
            rotary_base=source.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=getattr(source, "tie_word_embeddings", False),
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
        )

        return output

