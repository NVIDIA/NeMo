from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import torch.nn.functional as F

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo.lightning import io, teardown

if TYPE_CHECKING:
    from transformers import MistralConfig, MistralForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer


@dataclass
class MixtralConfig(GPTConfig):
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    gated_linear_unit: bool = True
    apply_query_key_layer_scaling: bool = False  # TODO: Should this be True?

    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: int = 8
    ffn_hidden_size: int = 14336
    max_position_embeddings: int = 4096  # 32768
    seq_length: int = 4096  # 32768
    # MoE
    num_moe_experts: int = 8
    moe_router_topk: int = 1

    init_method_std: float = 0.02
    layernorm_epsilon: float = 1e-5
    # rotary
    rotary_percent: float = 0.5
    rotary_base: float = 10000


class MixtralModel(GPTModel):
    def __init__(self, config: Optional[MixtralConfig] = None, optim_config=None, tokenizer=None):
        _tokenizer = tokenizer or HFMixtralImporter().tokenizer

        super().__init__(config or MixtralConfig(), optim_config, _tokenizer)


@io.model_importer(MixtralModel, ext="hf")
class HFMixtralImporter(io.ModelConnector["MixtralForCausalLM", MixtralModel]):
    def init(self) -> MixtralModel:
        return MixtralModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import MixtralForCausalLM

        source = MixtralForCausalLM.from_pretrained(str(self))
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        teardown(trainer, target)
        del trainer, target

        return output_path

    @property
    def tokenizer(self) -> "AutoTokenizer":
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(str(self))

    @property
    def config(self) -> MixtralConfig:
        from transformers import MixtralConfig as HfMixtralConfig

        config = HfMixtralConfig.from_pretrained(str(self))
        return MixtralConfig(
            activation_func=F.silu,
            # network
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,  # TODO
            seq_length=config.max_position_embeddings,
            # RoPE
            position_embedding_type='rope',
            rotary_base=source.rope_theta,
            # Transformer config
            num_attention_heads=config.num_attention_heads,
            num_query_groups=config.num_key_value_heads,
            num_moe_experts=config.num_local_experts,
            moe_router_topk=config.num_experts_per_tok,
            # norm
            normalization='rmsnorm',
            layernorm_epsilon=source.rms_norm_eps,
            # Init
            init_method_std=source.initializer_range,
            gated_linear_unit=True,
            # Vocab
            make_vocab_size_divisible_by=128,
        )
