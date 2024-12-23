from dataclasses import dataclass, field

import torch
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TELinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    param_is_not_tensor_parallel_duplicate,
)
from megatron.core.transformer.attention import CrossAttention, CrossAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType as MCoreAttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from torch import Tensor, nn

from nemo.collections.llm.gpt.model import local_layer_spec, transformer_engine_layer_spec


class ImageOutputProjectionPoolingHead(TransformerLayer):
    """Multihead Attention Pooling."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        num_query_token=77,
    ):
        super().__init__(config, submodules)

        self.probe = torch.nn.Parameter(torch.randn(num_query_token, 1, config.hidden_size))
        self.output_projection = build_module(
            submodules.output_linear_layer,
            input_size=4096,
            output_size=1024,
            config=config,
            bias=True,
            gather_output=True,
            skip_bias_add=False,
            is_expert=False,
            init_method=init_method_normal(0.02),
        )

    def forward(self, hidden_state):

        batch_size = hidden_state.shape[0]

        # [s, b, h]
        probe = self.probe.repeat(1, batch_size, 1)
        hidden_state = hidden_state.transpose(0, 1)
        hidden_state, context = super().forward(
            probe,
            attention_mask=None,
            context=hidden_state,
        )
        hidden_state, _ = self.output_projection(hidden_state)
        hidden_state = hidden_state.transpose(0, 1)
        return hidden_state
