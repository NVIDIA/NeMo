from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
    TENorm,
    TEColumnParallelLinear,
)
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from .spec_falcon_decoder_layer import FalconTransformerLayer, FalconTransformerLayerSubmodules

# Use this spec for an implementation using modules in TE
falcon_layer_spec = ModuleSpec(
    module=FalconTransformerLayer,
    submodules=FalconTransformerLayerSubmodules(
        input_layernorm=TENorm,
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=TEColumnParallelLinear,
                dot_product_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        post_self_attn_layernorm=TENorm,
        pre_mlp_layernorm=TENorm,
        mlp=ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear,
            ),
        ),
        mlp_bda=get_bias_dropout_add,
    ),
)
