# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Optional, Union

import torch
from megatron.core import parallel_state
from megatron.core.transformer.spec_utils import ModuleSpec
from torch import nn

from nemo.collections.llm.fn.activation import openai_gelu
from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo.collections.llm.utils import Config
from nemo.collections.nlp.models.language_modeling.megatron.gemma2.gemma2_spec import get_gemma2_layer_spec
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.pytorch.utils import dtype_from_hf

if TYPE_CHECKING:
    from transformers import GemmaForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


def gemma2_layer_spec(config: "GPTConfig") -> ModuleSpec:
    return get_gemma2_layer_spec()


# Note: Gemma requires huggingface transformers >= 4.38
# Note: these Gemma configs are copied from the corresponding HF model. You may need to modify the parameter for
# your own needs, in particular: seq_length and rotary_base.
@dataclass
class Gemma2Config(GPTConfig):
    # configs that are common across model sizes
    normalization: str = "RMSNorm"
    activation_func: Callable = openai_gelu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    seq_length: int = 8192
    kv_channels: int = 256
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = True
    layernorm_zero_centered_gamma: bool = True
    layernorm_epsilon: float = 1e-6
    rotary_base: float = 10000
    window_size: tuple = (4096, 0)
    vocab_size: int = 256000
    gradient_accumulation_fusion: bool = False

    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTConfig"], ModuleSpec]] = gemma2_layer_spec
    # mcore customization
    query_pre_attn_scalar: int = 224
    attn_logit_softcapping: float = 50.0
    final_logit_softcapping: float = 30.0


@dataclass
class Gemma2Config2B(Gemma2Config):
    num_layers: int = 26
    hidden_size: int = 2304
    num_attention_heads: int = 8
    num_query_groups: int = 4
    ffn_hidden_size: int = 9216
    query_pre_attn_scalar: int = 256


@dataclass
class Gemma2Config9B(Gemma2Config):
    num_layers: int = 42
    hidden_size: int = 3584
    num_attention_heads: int = 16
    num_query_groups: int = 8
    ffn_hidden_size: int = 14336
    query_pre_attn_scalar: int = 256


@dataclass
class Gemma2Config27B(Gemma2Config):
    num_layers: int = 46
    hidden_size: int = 4608
    num_attention_heads: int = 32
    num_query_groups: int = 16
    ffn_hidden_size: int = 36864
    query_pre_attn_scalar: int = 144


class Gemma2Model(GPTModel):
    def __init__(
        self,
        config: Annotated[Optional[Gemma2Config], Config[Gemma2Config]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config or Gemma2Config(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)

    def configure_model(self):
        from nemo.collections.common.parts.utils import extend_instance
        from nemo.collections.nlp.models.language_modeling.megatron.gemma2.gemma2_modules import Gemma2OutputLayer
        from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import EmbeddingScalingMixin

        super().configure_model()
        if parallel_state.is_pipeline_first_stage():
            # Apply Embedding Scaling: sqrt(hidden_size)
            extend_instance(self.module.embedding, EmbeddingScalingMixin)
        if parallel_state.is_pipeline_last_stage():
            # Prevents final logits from growing excessively by scaling them to a fixed range
            extend_instance(self.module.output_layer, Gemma2OutputLayer)


@io.model_importer(Gemma2Model, "hf")
class HFGemmaImporter(io.ModelConnector["GemmaForCausalLM", Gemma2Model]):
    def init(self) -> Gemma2Model:
        return Gemma2Model(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import Gemma2ForCausalLM

        source = Gemma2ForCausalLM.from_pretrained(str(self), torch_dtype='auto')
        target = self.init()

        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Gemma2 model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.pre_feedforward_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.layers.*.post_feedforward_layernorm.weight": "decoder.layers.*.mlp.linear_fc2.post_layernorm.weight",
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
        }

        return io.apply_transforms(source, target, mapping=mapping, transforms=[_import_qkv, _import_linear_fc1])

    @property
    def tokenizer(self) -> "AutoTokenizer":
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))

    @property
    def config(self) -> Gemma2Config:
        from transformers import GemmaConfig as HFGemmaConfig

        source = HFGemmaConfig.from_pretrained(str(self))

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        output = Gemma2Config(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
            num_query_groups=source.num_key_value_heads,
            rotary_base=source.rope_theta,
            query_pre_attn_scalar=source.query_pre_attn_scalar,
            attn_logit_softcapping=source.attn_logit_softcapping,
            final_logit_softcapping=source.final_logit_softcapping,
            window_size=(source.sliding_window, 0),
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            vocab_size=source.vocab_size,
            share_embeddings_and_output_weights=True,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
        )

        return output


@io.model_exporter(Gemma2Model, "hf")
class HFGemmaExporter(io.ModelConnector[Gemma2Model, "GemmaForCausalLM"]):
    def init(self) -> "GemmaForCausalLM":
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights(True):
            return AutoModelForCausalLM.from_config(self.config)

    def apply(self, output_path: Path) -> Path:
        target = self.init()
        source, _ = self.nemo_load(str(self))
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc2.post_layernorm.weight": "model.layers.*.post_feedforward_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
        }

        return io.apply_transforms(source, target, mapping=mapping, transforms=[_export_qkv, _export_linear_fc1])

    @property
    def tokenizer(self):
        return io.load_context(str(self)).model.tokenizer.tokenizer

    @property
    def config(self) -> "Gemma2Config":
        source: Gemma2Config = io.load_context(str(self)).model.config

        from transformers import Gemma2Config as HFGemmaConfig

        return HFGemmaConfig(
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            vocab_size=self.tokenizer.vocab_size,
            rope_theta=source.rotary_base,
            query_pre_attn_scalar=source.query_pre_attn_scalar,
            attn_logit_softcapping=source.attn_logit_softcapping,
            final_logit_softcapping=source.final_logit_softcapping,
        )


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
    head_size = megatron_config.kv_channels

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


@io.state_transform(
    source_key="model.layers.*.post_feedforward_layernorm.weight",
    target_key=(
        "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight",
        "decoder.layers.*.mlp.linear_fc2.post_layernorm.weight",
    ),
)
def _import_post_ffn_ln(ctx: io.TransformCTX, ln):
    return ln, ln


@io.state_transform(
    source_key="decoder.layers.*.self_attention.linear_qkv.weight",
    target_key=(
        "model.layers.*.self_attn.q_proj.weight",
        "model.layers.*.self_attn.k_proj.weight",
        "model.layers.*.self_attn.v_proj.weight",
    ),
)
def _export_qkv(ctx: io.TransformCTX, linear_qkv):
    megatron_config = ctx.source.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels
    qkv_total_dim = head_num + 2 * num_query_groups

    linear_qkv = linear_qkv.reshape([qkv_total_dim, head_size, hidden_size])
    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_proj = linear_qkv[q_slice].reshape(-1, hidden_size).cpu()
    k_proj = linear_qkv[k_slice].reshape(-1, hidden_size).cpu()
    v_proj = linear_qkv[v_slice].reshape(-1, hidden_size).cpu()

    return q_proj, k_proj, v_proj


@io.state_transform(
    source_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
    target_key="decoder.layers.*.mlp.linear_fc1.weight",
)
def _import_linear_fc1(down, gate):
    return torch.cat((down, gate), axis=0)


@io.state_transform(
    source_key="decoder.layers.*.mlp.linear_fc1.weight",
    target_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
)
def _export_linear_fc1(linear_fc1):
    gate_proj, up_proj = torch.chunk(linear_fc1, 2, dim=0)

    return gate_proj, up_proj


__all__ = [
    "Gemma2Config",
    "Gemma2Config2B",
    "Gemma2Config9B",
    "Gemma2Config27B",
    "Gemma2Model",
]