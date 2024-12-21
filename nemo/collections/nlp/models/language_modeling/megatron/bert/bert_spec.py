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

import warnings

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TENorm,
        TERowParallelLinear,
    )
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.spec_utils import ModuleSpec

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False
    from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults

    ModuleSpec = ApexGuardDefaults

from nemo.collections.nlp.models.language_modeling.megatron.bert.bert_model import (
    TransformerLayerSubmodulesWithPostLNSupport,
    TransformerLayerWithPostLNSupport,
)


# Use this spec to use lower level Transformer Engine modules (required for fp8 training)
def get_bert_layer_with_transformer_engine_spec_postln():
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).

    Returns:
        ModuleSpec: Module specification with TE modules
    """

    assert HAVE_MEGATRON_CORE, "Please ensure Megatron Core and Transformer Engine are installed."

    return ModuleSpec(
        module=TransformerLayerWithPostLNSupport,
        submodules=TransformerLayerSubmodulesWithPostLNSupport(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.padding},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            post_att_layernorm=TENorm,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
            post_mlp_layernorm=TENorm,
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_bert_layer_local_spec_postln():
    """Use this spec to use Megatron-Core spec.

    Returns:
        ModuleSpec: Module specification with Megatron-Core modules
    """
    return ModuleSpec(
        module=TransformerLayerWithPostLNSupport,
        submodules=TransformerLayerSubmodulesWithPostLNSupport(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.padding},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            post_att_layernorm=FusedLayerNorm,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear,
                    linear_fc2=RowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
            post_mlp_layernorm=FusedLayerNorm,
        ),
    )


def __getattr__(name):
    if name == 'bert_layer_with_transformer_engine_spec_postln':
        warnings.warn(
            """Attribute bert_specs.bert_layer_with_transformer_engine_spec_postln is on a
            deprecation track and will be removed in future releases. Please migrate to
            bert_spec.get_bert_layer_with_transformer_engine_spec_postln()."""
        )

        return get_bert_layer_with_transformer_engine_spec_postln()

    if name == 'bert_layer_local_spec_postln':
        warnings.warn(
            """Attribute bert_specs.bert_layer_local_spec_postln is on a
            deprecation track and will be removed in future releases. Please migrate to
            bert_spec.get_bert_layer_local_spec_postln()."""
        )

        return get_bert_layer_local_spec_postln()
