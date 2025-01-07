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

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TELinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.attention import CrossAttention, CrossAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType as MCoreAttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

from nemo.collections.multimodal.mimo.model.projection import ImageOutputProjectionPoolingHead


def get_output_projection_layer_spec() -> ModuleSpec:
    output_projection_submodules = TransformerLayerSubmodules(
        cross_attention=ModuleSpec(
            module=CrossAttention,
            params={"attn_mask_type": MCoreAttnMaskType.no_mask},
            submodules=CrossAttentionSubmodules(
                linear_q=TEColumnParallelLinear,
                linear_kv=TEColumnParallelLinear,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
            ),
        ),
        cross_attn_bda=get_bias_dropout_add,
        mlp=ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ),
        ),
        mlp_bda=get_bias_dropout_add,
    )
    output_projection_submodules.output_linear_layer = ColumnParallelLinear

    return ModuleSpec(module=ImageOutputProjectionPoolingHead, submodules=output_projection_submodules)
