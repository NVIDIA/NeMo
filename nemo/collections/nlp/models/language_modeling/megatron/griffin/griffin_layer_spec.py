# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from nemo.collections.nlp.models.language_modeling.megatron.griffin.recurrent_layer import RecurrentBlock, RecurrentBlockSubmodules
from nemo.collections.nlp.models.language_modeling.megatron.griffin.recurrent_module import RecurrentLayer, RecurrentLayerSubmodules
from nemo.collections.nlp.models.language_modeling.megatron.griffin.recurrent_module import Conv1D, RGLRU

griffin_mqa_layer_with_transformer_engine_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=TELayerNormColumnParallelLinear,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
                q_layernorm=IdentityOp,
                k_layernorm=IdentityOp,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        mlp=ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear, 
                linear_fc2=TERowParallelLinear,
            ),
        ),
        mlp_bda=get_bias_dropout_add,
    ),
)

griffin_recurrent_layer_with_transformer_engine_spec = ModuleSpec(
    module=RecurrentBlock,
    submodules=RecurrentBlockSubmodules(
        recurrent_layer=ModuleSpec(
            module=RecurrentLayer,
            submodules=RecurrentLayerSubmodules(
                linear_y=TELayerNormColumnParallelLinear,
                linear_x=TELayerNormColumnParallelLinear,
                linear_out=TERowParallelLinear,
                conv_1d=Conv1D,
                rg_lru=RGLRU,
            )
        ),
        recurrent_bda=get_bias_dropout_add,
        mlp=ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear, 
                linear_fc2=TERowParallelLinear,
            ),
        ),
        mlp_bda=get_bias_dropout_add,
    ),
)