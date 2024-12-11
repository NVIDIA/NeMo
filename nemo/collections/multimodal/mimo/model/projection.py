import torch
from torch import Tensor, nn


class TransformersProjector(nn.Module):
    def __init__(self, in_features, out_features, num_query_token, **kwargs):
        super().__init__()
        hidden_dim = 512
        self.in_fc = nn.Linear(in_features, hidden_dim)
        self.tfm = nn.Transformer(
            batch_first=True,
            norm_first=True,
            d_model=hidden_dim,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            nhead=4,
        )
        self.out_fc = nn.Linear(hidden_dim, out_features)

        self.query_embs = nn.Parameter(torch.randn(1, num_query_token, hidden_dim))
        self.query_embs.data.normal_(mean=0.0, std=0.0)

    def forward(self, x):
        # x = x + input_embs # Yash TODO: pass in input embeddings

        x = self.in_fc(x)
        x = self.tfm(x, self.query_embs.repeat(x.shape[0], 1, 1))
        if torch.distributed.get_rank() == 0:
            breakpoint()
        torch.distributed.barrier()
        outputs = self.out_fc(x)
        return outputs


from dataclasses import dataclass, field

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import CrossAttention, CrossAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType as MCoreAttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from nemo.collections.llm.gpt.model import local_layer_spec, transformer_engine_layer_spec


@dataclass
class Baseconfig(TransformerConfig):
    num_layers: int = 2
    num_attention_heads: int = 16  # was 32?
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    hidden_size: int = 4096
    hidden_dropout: int = 0.0
    attention_dropout: float = 0.0
    ffn_hidden_size: int = 4096
    gated_linear_unit: bool = False
    # activation_func = quick_gelu
    # kv_channels: int = 64
    # num_query_groups: int = 16
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False  # TODO: Yash Check this
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization = 'LayerNorm'
    layer_spec: ModuleSpec = transformer_engine_layer_spec


class TempPoolingHead(TransformerLayer):
    """Multihead Attention Pooling."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        num_query_token=77,
    ):
        super().__init__(config, submodules)

        self.probe = torch.nn.Parameter(torch.randn(num_query_token, 1, config.hidden_size))

    def forward(self, hidden_state):
        if torch.distributed.get_rank() == 0:
            breakpoint()
        torch.distributed.barrier()
        batch_size = hidden_state.shape[0]

        # [s, b, h]
        probe = self.probe.repeat(1, batch_size, 1)
        hidden_state = hidden_state.transpose(0, 1)
        hidden_state, context = super().forward(
            probe,
            attention_mask=None,
            context=hidden_state,
        )
        hidden_state = hidden_state.transpose(0, 1)
        if torch.distributed.get_rank() == 0:
            breakpoint()
        torch.distributed.barrier()
        return hidden_state
