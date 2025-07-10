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
from typing import Callable

import lightning.pytorch as L
import torch
from megatron.core.models.vision.radio import RADIOViTModel as MCoreRADIOViTModel

from nemo.collections.vlm.vision.layer_scaling import LayerScalingTransformerLayer, get_bias_dropout_add_layer_scaling

try:
    from megatron.core.extensions.transformer_engine import (
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )
except ImportError:
    from nemo.utils import logging

    # These Defaults are needed to make sure the code compiles
    TEDotProductAttention = object
    TERowParallelLinear = None
    TELayerNormColumnParallelLinear = None

    logging.warning(
        "Failed to import Transformer Engine dependencies. "
        "`from megatron.core.extensions.transformer_engine import *`"
        "If using NeMo Run, this is expected. Otherwise, please verify the Transformer Engine installation."
    )

from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

from nemo.collections.llm.fn.activation import openai_gelu
from nemo.collections.vlm.vision.base import CLIPViTConfig
from nemo.lightning import io, teardown


def get_norm_mlp_module_spec_te() -> ModuleSpec:
    """Get specs for MLP layer"""
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear),
    )


def get_radio_g_layer_spec_te() -> ModuleSpec:
    """Get specs for Radio transformer layer"""

    attn_mask_type = AttnMaskType.no_mask

    mlp = get_norm_mlp_module_spec_te()
    return ModuleSpec(
        module=LayerScalingTransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": attn_mask_type},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add_layer_scaling,
            pre_mlp_layernorm=IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add_layer_scaling,
        ),
    )


@dataclass
class RADIOViTConfig(CLIPViTConfig):
    """Intern ViT Base Config"""

    vision_model_type: str = "radio"
    patch_dim: int = 16
    img_h: int = 224
    img_w: int = 224
    max_img_h: int = 2048
    max_img_w: int = 2048
    class_token_len: int = 8
    max_num_tiles: int = 12
    num_layers: int = 32
    num_attention_heads: int = 16
    num_query_groups: int = 16
    kv_channels: int = 80
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    hidden_size: int = 1280
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    ffn_hidden_size: int = 5120
    gated_linear_unit: bool = False
    activation_func: Callable = openai_gelu
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization: str = 'LayerNorm'
    layernorm_epsilon: float = 1e-6
    apply_rope_fusion: bool = False
    embedder_bias: bool = False
    use_mask_token: bool = False
    use_thumbnail: bool = True

    def configure_model(self) -> "MCoreRADIOViTModel":
        # pylint: disable=C0115,C0116
        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            from nemo.collections.vlm.layer_specs import get_layer_spec_te

            if "raido_g" in self.vision_model_type:
                transformer_layer_spec = get_layer_spec_te()
            else:
                transformer_layer_spec = get_layer_spec_te(is_vit=True)
        return MCoreRADIOViTModel(
            self,
            transformer_layer_spec,
            img_h=self.img_h,
            img_w=self.img_w,
            max_img_h=self.max_img_h,
            max_img_w=self.max_img_w,
            class_token_len=self.class_token_len,
            patch_dim=self.patch_dim,
            add_class_token=self.add_class_token,
            embedder_bias=self.embedder_bias,
            use_mask_token=self.use_mask_token,
        )


@dataclass
class RADIO_25_h_Config(RADIOViTConfig):
    """Radio v2.5 h Config"""

    vision_model_type: str = "radio"


@dataclass
class RADIO_25_g_Config(RADIOViTConfig):
    """Radio v2.5 g Config"""

    vision_model_type: str = "raido_g"
    num_layers: int = 40
    num_attention_heads: int = 24
    num_query_groups: int = 24
    kv_channels: int = 64
    hidden_size: int = 1536
    ffn_hidden_size: int = 4096
    activation_func: Callable = torch.nn.functional.silu


class RADIOViTModel(L.LightningModule, io.IOMixin, io.ConnectorMixin):
    """RADIOViT NeMo Wrapper"""

    def __init__(self, config):
        # pylint: disable=C0115,C0116
        super().__init__()
        self.config = config

    def configure_model(self) -> None:
        # pylint: disable=C0115,C0116
        if not hasattr(self, "module"):
            self.module = self.config.configure_model()


@io.model_importer(RADIOViTModel, "hf")
class HFRADIOViTImporter(io.ModelConnector["RADIOViTModel", RADIOViTModel]):
    """HF RADIOViT Importer"""

    def init(self) -> RADIOViTModel:
        # pylint: disable=C0115,C0116
        return RADIOViTModel(self.config)

    def apply(self, output_path: Path) -> Path:
        # pylint: disable=C0115,C0116
        from transformers import AutoModel

        source = AutoModel.from_pretrained(str(self), trust_remote_code=True)
        target = self.init()
        trainer = self.nemo_setup(target)

        self.convert_state(source, target)
        print(f"Converted RADIOViT model to Nemo, saving to {output_path}")

        self.nemo_save(output_path, trainer)

        print(f"Converted RADIOViT model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        # pylint: disable=C0115,C0116
        mapping = {
            # Embeddings
            "embedder": "embedder.weight",
            "cls_token": "class_token",
            "pos_embed": "position_embeddings",
            # Transformer Layers
            "blocks.*.attn.qkv.weight": "decoder.layers.*.self_attention.linear_qkv.weight",
            "blocks.*.attn.qkv.bias": "decoder.layers.*.self_attention.linear_qkv.bias",
            "blocks.*.attn.proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "blocks.*.attn.proj.bias": "decoder.layers.*.self_attention.linear_proj.bias",
            # Layer norms
            "blocks.*.norm1.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "blocks.*.norm1.bias": "decoder.layers.*.self_attention.linear_qkv.layer_norm_bias",
            "blocks.*.norm2.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "blocks.*.norm2.bias": "decoder.layers.*.mlp.linear_fc1.layer_norm_bias",
            # MLP
            "blocks.*.mlp.fc1.weight": "decoder.layers.*.mlp.linear_fc1.weight",
            "blocks.*.mlp.fc1.bias": "decoder.layers.*.mlp.linear_fc1.bias",
            "blocks.*.mlp.fc2.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "blocks.*.mlp.fc2.bias": "decoder.layers.*.mlp.linear_fc2.bias",
        }

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=[_import_qkv, _import_qkv_bias],
        )

    @property
    def config(self) -> CLIPViTConfig:
        # pylint: disable=C0115,C0116
        # TODO(yuya): add config converting
        output = RADIOViTConfig()
        return output


def import_qkv(q, k, v, head_num, num_query_groups, heads_per_group, hidden_size, head_size):
    # pylint: disable=C0115,C0116
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
    source_key="blocks.*.attn.qkv.weight",
    target_key="decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_qkv(ctx: io.TransformCTX, qkv):
    # pylint: disable=C0115,C0116
    megatron_config = ctx.target.config
    q, k, v = qkv.chunk(3)
    return import_qkv(
        q,
        k,
        v,
        head_num=megatron_config.num_attention_heads,
        num_query_groups=megatron_config.num_query_groups,
        heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
        hidden_size=megatron_config.hidden_size,
        head_size=megatron_config.kv_channels,
    )


@io.state_transform(
    source_key="blocks.*.attn.qkv.bias",
    target_key="decoder.layers.*.self_attention.linear_qkv.bias",
)
def _import_qkv_bias(ctx: io.TransformCTX, qkv_bias):
    # pylint: disable=C0115,C0116
    megatron_config = ctx.target.config
    q_bias, k_bias, v_bias = qkv_bias.chunk(3)
    return import_qkv(
        q_bias.unsqueeze(-1),
        k_bias.unsqueeze(-1),
        v_bias.unsqueeze(-1),
        head_num=megatron_config.num_attention_heads,
        num_query_groups=megatron_config.num_query_groups,
        heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
        hidden_size=1,
        head_size=megatron_config.kv_channels,
    ).squeeze(-1)
