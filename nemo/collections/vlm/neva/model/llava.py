from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Union, Optional

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.llm import Llama2Config7B, Llama2Config13B, Llama2Config70B
from nemo.collections.vlm.neva.model.base import NevaConfig, NevaModel
from nemo.collections.vlm.neva.model.base import HFCLIPVisionConfig, CLIPViTConfig, MultimodalProjectorConfig
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown

from megatron.core.transformer.transformer_config import TransformerConfig

if TYPE_CHECKING:
    from transformers import LlavaConfig as HFLlavaConfig
    from transformers import LlavaForConditionalGeneration

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


# Note: these Llava configs are copied from the corresponding HF model. You may need to modify the parameter for
# your own needs, in particular: seq_length and rotary_base.

@dataclass
class LlavaConfig(NevaConfig):
    drop_vision_class_token: bool = True


@dataclass
class Llava1_5Config7B(LlavaConfig):
    from transformers import PretrainedConfig
    language_transformer_config: TransformerConfig = Llama2Config7B()
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = HFCLIPVisionConfig(
        pretrained_model_name_or_path="openai/clip-vit-large-patch14-336"
    )
    vision_projection_config: TransformerConfig = MultimodalProjectorConfig(
        input_size=1024)


# @dataclass
# class Llava1_5Config13B(LlavaConfig):
#     num_layers: int = 40
#     hidden_size: int = 5120
#     num_attention_heads: int = 40
#     num_query_groups: int = 40
#     ffn_hidden_size: int = 13824
#
#
# @dataclass
# class Llava1_5Config70B(LlavaConfig):
#     num_layers: int = 80
#     hidden_size: int = 8192
#     num_attention_heads: int = 64
#     num_query_groups: int = 8
#     ffn_hidden_size: int = 28672


class LlavaModel(NevaModel):
    def __init__(
        self,
        config: Annotated[Optional[LlavaConfig], Config[LlavaConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config or LlavaConfig(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)


@io.model_importer(LlavaModel, "hf")
class HFLlavaImporter(io.ModelConnector["LlavaForConditionalGeneration", LlavaModel]):
    def init(self) -> LlavaModel:
        return LlavaModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import LlavaForConditionalGeneration

        source = LlavaForConditionalGeneration.from_pretrained(str(self))
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Llava model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "language_model.model.embed_tokens.weight": "language_model.embedding.word_embeddings.weight",
            "language_model.model.layers.*.self_attn.o_proj.weight": "language_model.decoder.layers.*.self_attention.linear_proj.weight",
            "language_model.model.layers.*.mlp.down_proj.weight": "language_model.decoder.layers.*.mlp.linear_fc2.weight",
            "language_model.model.layers.*.input_layernorm.weight": "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "language_model.model.layers.*.post_attention_layernorm.weight": "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "language_model.model.norm.weight": "language_model.decoder.final_layernorm.weight",
            "language_model.lm_head.weight": "language_model.output_layer.weight",
            "vision_tower.vision_model.*": "vision_model.vision_model.*",
        }

        return io.apply_transforms(source, target, mapping=mapping, transforms=[_import_qkv, _import_linear_fc1])

    @property
    def tokenizer(self) -> "AutoTokenizer":
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(str(self))

    @property
    def config(self) -> LlavaConfig:
        from transformers import LlavaConfig as HFLlavaConfig

        source = HFLlavaConfig.from_pretrained(str(self))

        # def make_vocab_size_divisible_by(vocab_size):
        #     base = 128
        #     while vocab_size % base != 0:
        #         base //= 2
        #     return base

        output = Llava1_5Config7B()
        output.language_transformer_config.make_vocab_size_divisible_by = 64

        return output


@io.model_exporter(LlavaModel, "hf")
class HFLlavaExporter(io.ModelConnector[LlavaModel, "LlavaForConditionalGeneration"]):
    def init(self) -> "LlavaForConditionalGeneration":
        from transformers import AutoModelForCausalLM

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
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }

        return io.apply_transforms(source, target, mapping=mapping, transforms=[_export_qkv, _export_linear_fc1])

    @property
    def tokenizer(self):
        return io.load_context(str(self)).model.tokenizer.tokenizer

    @property
    def config(self) -> "HFLlavaConfig":
        source: LlavaConfig = io.load_context(str(self)).model.config

        from transformers import LlavaConfig as HFLlavaConfig

        return HFLlavaConfig(
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            vocab_size=self.tokenizer.vocab_size,
        )


@io.state_transform(
    source_key=(
        "language_model.model.layers.*.self_attn.q_proj.weight",
        "language_model.model.layers.*.self_attn.k_proj.weight",
        "language_model.model.layers.*.self_attn.v_proj.weight",
    ),
    target_key="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_qkv(ctx: io.TransformCTX, q, k, v):
    megatron_config = ctx.target.config.language_transformer_config
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
    head_num = megatron_config.num_attention_heads
    head_size = hidden_size // head_num
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
    source_key=("language_model.model.layers.*.mlp.gate_proj.weight", "language_model.model.layers.*.mlp.up_proj.weight"),
    target_key="language_model.decoder.layers.*.mlp.linear_fc1.weight",
)
def _import_linear_fc1(down, gate):
    return torch.cat((down, gate), axis=0).float()


@io.state_transform(
    source_key="decoder.layers.*.mlp.linear_fc1.weight",
    target_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
)
def _export_linear_fc1(linear_fc1):
    gate_proj, up_proj = torch.chunk(linear_fc1, 2, dim=0)

    return gate_proj, up_proj