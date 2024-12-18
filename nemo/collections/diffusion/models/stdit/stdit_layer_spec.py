import copy
from dataclasses import dataclass
from typing import Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from megatron.core.jit import jit_fuser
from megatron.core.parallel_state import get_context_parallel_group, get_context_parallel_world_size
from megatron.core.tensor_parallel import all_to_all
from megatron.core.transformer.attention import (
    CrossAttention,
    CrossAttentionSubmodules,
    SelfAttention,
    SelfAttentionSubmodules,
)
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import make_viewless_tensor

from .spatial_selfattention_util import SpatialSelfAttention


# Todo : implemention of stditv3 t_select_mask
@dataclass
class STDiTV3LayerWithAdaLNSubmodules(TransformerLayerSubmodules):
    """
    STDiTV3_Layer: spatial block & temporal block
    """

    spatial_self_attention: Union[ModuleSpec, type] = IdentityOp
    temporal_self_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attention_spatial: Union[ModuleSpec, type] = IdentityOp
    cross_attention_temporal: Union[ModuleSpec, type] = IdentityOp
    mlp_spatial: Union[ModuleSpec, type] = IdentityOp
    mlp_temporal: Union[ModuleSpec, type] = IdentityOp


# dynamic parallel need alltoall comm
def all_to_all_temporal_p2spatial_p(input_):
    """
    Perform AlltoAll communication on context parallel group, transform the input tensor from shape [T/CP, HW, B, D] to [T, HW/CP, B, D].

    Args:
        input_ (torch.Tensor): The input tensor which hash been distributed along Temporal dimension with shape [T/CP, HW, B, D].

    Returns:
        torch.Tensor: The output tensor with shape [T, HW/CP, B, D].
    """
    world_size = get_context_parallel_world_size()
    cp_group = get_context_parallel_group()
    split_tensors = torch.split(input_, split_size_or_sections=input_.shape[1] // world_size, dim=1)
    concat_tensor = torch.cat(split_tensors, dim=0)
    output = all_to_all(cp_group, concat_tensor)
    return output


def all_to_all_spatial_p2temporal_p(input_):
    """
    Perform AlltoAll communication on context parallel group, transform the input tensor from shape [T, HW/CP, B, D] to [T/CP, HW, B, D].

    Args:
        input_ (torch.Tensor): The input tensor which hash been distributed along Temporal dimension with shape [T, HW/CP, B, D].

    Returns:
        torch.Tensor: The output tensor with shape [T/CP, HW, B, D].
    """
    world_size = get_context_parallel_world_size()
    cp_group = get_context_parallel_group()
    input_exchanged = all_to_all(cp_group, input_)
    split_tensors = torch.split(input_exchanged, split_size_or_sections=input_exchanged.shape[0] // world_size, dim=0)
    output = torch.cat(split_tensors, dim=1).contiguous()
    return output


## reference
## megatron implementation: https://github.com/NVIDIA/Megatron-LM/blob/0363328d21c40c98d816847ebbbd9bfd84b50f1e/megatron/legacy/model/transformer.py#L60
## timm implementation: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py#L150


class DropPath(MegatronModule):
    """
    Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_state):
        if self.drop_prob == 0.0 or not self.training:
            return hidden_state
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        # hidden_state: [s, b, h]
        shape = (1,) + (hidden_state.shape[1],) + (1,) * (hidden_state.ndim - 2)
        random_tensor = keep_prob + torch.rand(shape, dtype=hidden_state.dtype, device=hidden_state.device)
        random_tensor.floor_()  # binarize
        output = hidden_state.div(keep_prob) * random_tensor
        return output


class AdaLN_STDiTv3(MegatronModule):
    """
    Adaptive Layer Normalization Module for STDiT
    For open-sora part adaLN_modulation diff with origin DiT model in forward function here.
    """

    def __init__(self, config: TransformerConfig, n_adaln_chunks=6):
        super().__init__(config)
        self.ln = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)
        self.n_adaln_chunks = n_adaln_chunks
        self.scale_shift_table = nn.Parameter(
            torch.randn(self.n_adaln_chunks, config.hidden_size) / config.hidden_size**0.5
        )
        setattr(self.scale_shift_table, "sequence_parallel", config.sequence_parallel)

    def forward(self, timestep_emb, batch):
        # [Batch, Chunk, Dimension] -> [Chunk, Batch, Dimension]
        return (
            rearrange(
                self.scale_shift_table[None] + timestep_emb.reshape(batch, self.n_adaln_chunks, -1), "B C D -> C B D"
            )
        ).chunk(self.n_adaln_chunks, dim=0)

    @jit_fuser
    def scale_add(self, residual, x, gate):
        return residual + gate * x

    @jit_fuser
    def scale_add_bias(self, residual, x, x_bias, gate):
        return residual + gate * (x + x_bias)

    @jit_fuser
    def residual_add_bias(self, residual, x, x_bias):
        return residual + x + x_bias

    @jit_fuser
    def modulate(self, x, shift, scale):
        return x * (1 + scale) + shift

    @jit_fuser
    def modulated_layernorm(self, x, shift, scale):
        # Optional Input Layer norm
        input_layernorm_output = self.ln(x).type_as(x)

        # DiT block specific
        return self.modulate(input_layernorm_output, shift, scale)

    @jit_fuser
    def add_bias_modulated_layernorm(self, residual, x, x_bias, shift, scale):
        hidden_states = self.residual_add_bias(residual, x, x_bias)
        pre_mlp_ada_norm_output = self.modulated_layernorm(hidden_states, shift, scale)
        return hidden_states, pre_mlp_ada_norm_output


class STDiTV3LayerWithAdaLN(TransformerLayer):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        drop_path_rate: float = 0.0,
        mlp_ratio: float = 4.0,
    ):

        config.ffn_hidden_size = int(mlp_ratio * config.hidden_size)

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
        )

        ## [Module 1: Ada Layernorm before self_attention]
        self.spatial_adaln_stdit = AdaLN_STDiTv3(config=self.config, n_adaln_chunks=6)
        self.temporal_adaln_stdit = AdaLN_STDiTv3(config=self.config, n_adaln_chunks=6)

        ## [Module 2: SpatialSelfAttention TemporalselfAttention]
        self_attention_config = copy.deepcopy(self.config)
        self_attention_config.add_qkv_bias = True
        self_attention_config.qk_layernorm_per_head = True
        self_attention_config.normalization = "RMSNorm"

        self_attention_spatial_config = copy.deepcopy(self_attention_config)
        if self.config.dynamic_sequence_parallel:
            self_attention_spatial_config.context_parallel_size = 1

        self_attention_temporal_config = copy.deepcopy(self_attention_config)
        self_attention_temporal_config.context_parallel_size = 1

        self.spatial_self_attention = build_module(
            submodules.spatial_self_attention,
            config=self_attention_spatial_config,
            layer_number=layer_number,
        )

        self.temporal_self_attention = build_module(
            submodules.temporal_self_attention,
            config=self_attention_temporal_config,
            layer_number=layer_number,
        )

        ## [Module 3: CrossAttention]
        cross_attention_config = copy.deepcopy(self.config)
        cross_attention_config.qk_layernorm_cross_attn = False
        cross_attention_config.qk_layernorm_per_head = True
        cross_attention_config.context_parallel_size = 1

        self.cross_attention_spatial = build_module(
            submodules.cross_attention_spatial,
            config=cross_attention_config,
            layer_number=layer_number,
        )
        self.cross_attention_temporal = build_module(
            submodules.cross_attention_temporal,
            config=cross_attention_config,
            layer_number=layer_number,
        )

        ## [Module 4: MLP]
        mlp_config = copy.deepcopy(self.config)
        mlp_config.activation_func = F.gelu
        # Todo : check mlp_activation gelu(approximate="tanh")
        # mlp_config.activation_func = lambda x: F.gelu(x, approximate="tanh")

        self.mlp_spatial = build_module(
            submodules.mlp_spatial,
            config=mlp_config,
        )
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        self.mlp_temporal = build_module(
            submodules.mlp_temporal,
            config=mlp_config,
        )
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        ## [Module 5: DropPath] DropPath
        self.drop_path_spatial = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None
        self.drop_path_temporal = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

    def dynamic_sequence_parallel_comm(self, x, T, S, temporal_p2spatial_p: bool):
        """
        dynamic sequence parallel part : commmunication all2all between cp_group
        x : input shape [S, B, D]
        T : temporal dim
        S : spatial dim
        temporal_p2spatial_p
            - True  : all2all from [temporal/cp, spatial] to [temporal, spatial/cp]
            - False : all2all from [temporal, spatial/cp] to [temporal/cp, spatial]
        """
        x = rearrange(x, "(T S) B D -> T S B D", T=T, S=S)

        if temporal_p2spatial_p:
            x = all_to_all_temporal_p2spatial_p(x)
        else:
            x = all_to_all_spatial_p2temporal_p(x)

        dsp_t, dsp_s = x.shape[0], x.shape[1]
        x = rearrange(x, "T S B D -> (T S) B D", T=dsp_t, S=dsp_s)

        return x, dsp_t, dsp_s

    def forward(
        self,
        hidden_states,  # noise_latent | x [S, B, D]
        attention_mask,  # time_steps, time_embedding | t [B, D]
        context=None,  # context, text_embedding | y [S, B, D]
        context_mask=None,  # cross attention mask
        rotary_pos_emb=None,  # rotary_pos_embeddin
        inference_params=None,
        packed_seq_params=None,
    ):

        # timestep embedding
        timestep_emb = attention_mask

        Batch = hidden_states.size(1)

        # ------------------------------------------ spatial block  -----------------------------------------------#

        # random_set_shift_scale
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = self.spatial_adaln_stdit(timestep_emb, Batch)

        # adaln
        pre_spatial_attn_ada_norm_output = self.spatial_adaln_stdit.modulated_layernorm(
            hidden_states, shift=shift_sa, scale=scale_sa
        )

        # *************************** spatial self attention **************************

        # alltoall from spatial/cp to temporal/cp
        if self.config.dynamic_sequence_parallel:
            pre_spatial_attn_ada_norm_output, temporal_dim, spatial_dim = self.dynamic_sequence_parallel_comm(
                pre_spatial_attn_ada_norm_output,
                T=self.config.stdit_dim_T,
                S=self.config.stdit_dim_S,
                temporal_p2spatial_p=False,
            )
        else:
            temporal_dim = self.config.stdit_dim_T
            spatial_dim = self.config.stdit_dim_S

        # spatial attention, reshape in selfattention using spatial type
        spatial_attention_output, spatial_attention_output_bias = self.spatial_self_attention(
            pre_spatial_attn_ada_norm_output,
            attention_mask=None,
        )

        # add bias here
        spatial_attention_output = spatial_attention_output + spatial_attention_output_bias

        # alltoall from temporal/cp to spatial/cp
        if self.config.dynamic_sequence_parallel:
            spatial_attention_output, temporal_dim, spatial_dim = self.dynamic_sequence_parallel_comm(
                spatial_attention_output, T=temporal_dim, S=spatial_dim, temporal_p2spatial_p=True
            )

        # gate scale
        # residual + drop_out_path
        if self.drop_path_spatial is None:
            hidden_states = self.spatial_adaln_stdit.scale_add(hidden_states, spatial_attention_output, gate_sa)
        else:
            hidden_states = hidden_states + self.drop_path_spatial(spatial_attention_output * gate_sa)

        # ***************************  cross attention  *******************************

        cross_attention_output, cross_attention_output_bias = self.cross_attention_spatial(
            hidden_states,
            attention_mask=context_mask,
            key_value_states=context,
            packed_seq_params=packed_seq_params,
        )

        hidden_states, pre_mlp_ada_norm_output = self.spatial_adaln_stdit.add_bias_modulated_layernorm(
            hidden_states, cross_attention_output, cross_attention_output_bias, shift=shift_mlp, scale=scale_mlp
        )

        # *****************************    mlp   *******************************
        mlp_output, mlp_output_bias = self.mlp_spatial(pre_mlp_ada_norm_output)

        # gate scale
        # residual + drop_out_path
        if self.drop_path_spatial is None:
            hidden_states = self.spatial_adaln_stdit.scale_add_bias(
                hidden_states, mlp_output, mlp_output_bias, gate_mlp
            )
        else:
            hidden_states = hidden_states + self.drop_path_spatial((mlp_output + mlp_output_bias) * gate_mlp)

        # ------------------------------------------ temporal block  -----------------------------------------------#

        # random_set_shift_scale
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = self.temporal_adaln_stdit(timestep_emb, Batch)

        # adaln
        pre_temporal_attn_ada_norm_output = self.temporal_adaln_stdit.modulated_layernorm(
            hidden_states, shift=shift_sa, scale=scale_sa
        )

        # ************************** temoral self attention ***************************

        # rearrange format
        pre_temporal_attn_ada_norm_output = rearrange(
            pre_temporal_attn_ada_norm_output,
            "(T S) B D -> T (B S) D",
            T=self.config.stdit_dim_T,
            S=self.config.stdit_dim_S,
        )

        # attention
        temporal_attention_output, temporal_attention_output_bias = self.temporal_self_attention(
            pre_temporal_attn_ada_norm_output, attention_mask=None, rotary_pos_emb=rotary_pos_emb
        )

        # rearrange format
        temporal_attention_output = rearrange(
            temporal_attention_output + temporal_attention_output_bias,
            "T (B S) D -> (T S) B D",
            T=self.config.stdit_dim_T,
            S=self.config.stdit_dim_S,
        )

        # gate scale
        # residual + drop_out_path
        if self.drop_path_temporal is None:
            hidden_states = self.temporal_adaln_stdit.scale_add(hidden_states, temporal_attention_output, gate_sa)
        else:
            hidden_states = hidden_states + self.drop_path_temporal(temporal_attention_output * gate_sa)

        # ***************************  cross attention  *******************************
        cross_attention_output, cross_attention_output_bias = self.cross_attention_temporal(
            hidden_states,
            attention_mask=context_mask,
            key_value_states=context,
            packed_seq_params=packed_seq_params,
        )

        hidden_states, pre_mlp_ada_norm_output = self.temporal_adaln_stdit.add_bias_modulated_layernorm(
            hidden_states, cross_attention_output, cross_attention_output_bias, shift=shift_mlp, scale=scale_mlp
        )

        # *****************************    mlp   *******************************
        mlp_output, mlp_output_bias = self.mlp_temporal(pre_mlp_ada_norm_output)

        # gate scale
        # residual + drop_out_path
        if self.drop_path_temporal is None:
            hidden_states = self.temporal_adaln_stdit.scale_add_bias(
                hidden_states, mlp_output, mlp_output_bias, gate_mlp
            )
        else:
            hidden_states = hidden_states + self.drop_path_temporal((mlp_output + mlp_output_bias) * gate_mlp)

        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)

        return output, context


def get_stdit_analn_block_with_transformer_engine_spec() -> ModuleSpec:
    """T5 decoder TE spec (uses Transformer Engine components)."""
    # init_params cross attention
    # pack : padding & unpacked : no_mask
    # from megatron.training import get_args
    # args = get_args()
    # params = {"attn_mask_type": AttnMaskType.padding if args.packing_algorithm == 'pack_format_sbhd' else AttnMaskType.no_mask}
    params = {"attn_mask_type": AttnMaskType.no_mask}
    return ModuleSpec(
        module=STDiTV3LayerWithAdaLN,
        submodules=STDiTV3LayerWithAdaLNSubmodules(
            spatial_self_attention=ModuleSpec(
                module=SpatialSelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                ),
            ),
            cross_attention_spatial=ModuleSpec(
                module=CrossAttention,
                params=params,
                submodules=CrossAttentionSubmodules(
                    linear_q=TEColumnParallelLinear,
                    linear_kv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,  # in stditv3, no q_layernorm in cross_attn
                    k_layernorm=IdentityOp,  # in stditv3, no k_layernorm in cross_attn
                ),
            ),
            mlp_spatial=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
            temporal_self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                ),
            ),
            cross_attention_temporal=ModuleSpec(
                module=CrossAttention,
                params=params,
                submodules=CrossAttentionSubmodules(
                    linear_q=TEColumnParallelLinear,
                    linear_kv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,  # in stditv3, no q_layernorm in cross_attn
                    k_layernorm=IdentityOp,  # in stditv3, no k_layernorm in cross_attn
                ),
            ),
            mlp_temporal=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
        ),
    )
