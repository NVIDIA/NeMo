from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown
from nemo.collections.nlp.modules.common.megatron.utils import squared_relu

if TYPE_CHECKING:
    from transformers import NemotronConfig as HFNemotronConfig
    from transformers import NemotronForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec



@dataclass
class NemotronConfig(GPTConfig):
    # configs that are common across model sizes
    normalization: str = "LayerNorm"
    activation_func: Callable = squared_relu
    add_bias_linear: bool = False
    seq_length: int = 4096
    position_embedding_type: str = "rope"
    rotary_percent: float = 0.5
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    layernorm_zero_centered_gamma: bool = True # layernorm1p
    init_method_std: float = 0.01
    # apply_query_key_layer_scaling: bool = True
    share_embeddings_and_output_weights: bool = False
    

@dataclass
class Nemotron3Config8B(NemotronConfig):
    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 16384
    num_attention_heads: int = 32
    

class NemotronModel(GPTModel):
    def __init__(
        self,
        config: Annotated[Optional[NemotronConfig], Config[NemotronConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config or NemotronConfig(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)

@io.model_importer(NemotronModel, "hf")
class HFNemotronImporter(io.ModelConnector["NemotronForCausalLM", NemotronModel]):
    def init(self) -> NemotronModel:
        return NemotronModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import NemotronForCausalLM

        source = NemotronForCausalLM.from_pretrained(str(self))
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Nemotron model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.mlp.up_proj.weight": "decoder.layers.*.mlp.linear_fc1.weight",
            "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
            "lm_head.weight": "output_layer.weight",
        }

        return io.apply_transforms(source, target, mapping=mapping, transforms=[_import_qkv])

    @property
    def tokenizer(self) -> "AutoTokenizer":
        # from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
        # return get_nmt_tokenizer(model_name='/aot/checkpoints/nemotron/nemotron3-8b/tokenizer.model', 
        #                           tokenizer_model='/aot/checkpoints/nemotron/nemotron3-8b/tokenizer.model')
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(str(self))

    @property
    def config(self) -> NemotronConfig:
        from transformers import NemotronConfig as HFNemotronConfig

        source = HFNemotronConfig.from_pretrained(str(self))

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        output = NemotronConfig(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            seq_length=source.max_position_embeddings,
            layernorm_epsilon=source.norm_eps,
            num_query_groups=source.num_key_value_heads,
            rotary_base=source.rope_theta,
            rotary_percent=source.partial_rotary_factor,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=False,
        )

        return output


@io.state_transform(
    source_key=(
        "model.layers.*.self_attn.q_proj.weight",
        "model.layers.*.self_attn.k_proj.weight",
        "model.layers.*.self_attn.v_proj.weight",
    ),
    target_key="decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_qkv(ctx: io.TransformCTX, q, k, v):
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_num = megatron_config.num_attention_heads
    head_size = hidden_size // head_num

    old_tensor_shape = q.size()
    new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
    new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]

    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_kv_tensor_shape)
    v = v.view(*new_kv_tensor_shape)

    qkv_weights_l = []
    for i in range(num_query_groups):
        qkv_weights_l.append(q[i * heads_per_group : (i + 1) * heads_per_group, :, :])
        qkv_weights_l.append(k[i : i + 1, :, :])
        qkv_weights_l.append(v[i : i + 1, :, :])
    qkv_weights = torch.cat(qkv_weights_l)
    assert qkv_weights.ndim == 3, qkv_weights.shape
    assert qkv_weights.shape[0] == (heads_per_group + 2) * num_query_groups, qkv_weights.shape
    assert qkv_weights.shape[1] == head_size, qkv_weights.shape
    assert qkv_weights.shape[2] == old_tensor_shape[1], qkv_weights.shape

    qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])

    return qkv_weights
