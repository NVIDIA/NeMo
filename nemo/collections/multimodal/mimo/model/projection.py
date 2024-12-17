import torch
from torch import Tensor, nn
from megatron.core.transformer.spec_utils import build_module
from megatron.core.utils import init_method_normal, scaled_init_method_normal
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
    TELinear
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
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
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
    output_projection_submodules.output_linear_layer = ColumnParallelLinear #TEColumnParallelLinear 
    
    return ModuleSpec(module =TempPoolingHead, submodules=output_projection_submodules )

from megatron.core.tensor_parallel.layers import param_is_not_tensor_parallel_duplicate

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
        self.output_projection = build_module(submodules.output_linear_layer, input_size=4096, output_size=1024, config = config, bias = True,gather_output=True, skip_bias_add= False, is_expert=False,init_method = init_method_normal(0.02))
        
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
