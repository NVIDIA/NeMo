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

# pylint: disable=C0115,C0116,C0301

import copy
from dataclasses import dataclass
from typing import Literal, Union

import torch.nn as nn
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


@dataclass
class DiTWithAdaLNSubmodules(TransformerLayerSubmodules):
    temporal_self_attention: Union[ModuleSpec, type] = IdentityOp
    full_self_attention: Union[ModuleSpec, type] = IdentityOp


class AdaLN(MegatronModule):
    """
    Adaptive Layer Normalization Module for DiT.
    """

    def __init__(
        self, config: TransformerConfig, n_adaln_chunks=9, use_adaln_lora=True, adaln_lora_dim=256, norm=nn.LayerNorm
    ):
        super().__init__(config)
        if norm == TENorm:
            self.ln = norm(config, config.hidden_size, config.layernorm_epsilon)
        else:
            self.ln = norm(config.hidden_size, elementwise_affine=False, eps=self.config.layernorm_epsilon)
        self.n_adaln_chunks = n_adaln_chunks
        if use_adaln_lora:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(config.hidden_size, adaln_lora_dim * 3, bias=False),
                nn.Linear(adaln_lora_dim * 3, self.n_adaln_chunks * config.hidden_size, bias=False),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(config.hidden_size, self.n_adaln_chunks * config.hidden_size, bias=False)
            )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)

        setattr(self.adaLN_modulation[-1].weight, "sequence_parallel", config.sequence_parallel)

    def forward(self, timestep_emb, adaln_lora_B_3D):
        adaln_lora_B_3D = list(adaln_lora_B_3D.chunk(3, dim=-1))
        modulation = list((self.adaLN_modulation(timestep_emb)).chunk(self.n_adaln_chunks, dim=-1))
        for i in range(len(modulation)):
            modulation[i] = modulation[i] + adaln_lora_B_3D[i % 3]
        return modulation

    # @jit_fuser
    def modulate(self, x, shift, scale):
        return x * (1 + scale) + shift

    # @jit_fuser
    def scale_add(self, residual, x, gate):
        return residual + gate * x

    # @jit_fuser
    def modulated_layernorm(self, x, shift, scale):
        # Optional Input Layer norm
        input_layernorm_output = self.ln(x).type_as(x)

        # z = rearrange(input_layernorm_output, '(T H W) B D -> B T H W D', T=4, H=16, W=16)
        # print(f'megatron after norm = {z}, shape={z.shape}')

        # DiT block specific
        return self.modulate(input_layernorm_output, shift, scale)

    # @jit_fuser
    def scaled_modulated_layernorm(self, residual, x, gate, shift, scale):
        hidden_states = self.scale_add(residual, x, gate)
        shifted_pre_mlp_layernorm_output = self.modulated_layernorm(hidden_states, shift, scale)
        return hidden_states, shifted_pre_mlp_layernorm_output


class DiTLayerWithAdaLN(TransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.

    DiT with Adapative Layer Normalization.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute",
    ):
        def _replace_no_cp_submodules(submodules):
            modified_submods = copy.deepcopy(submodules)
            modified_submods.cross_attention = IdentityOp
            # modified_submods.temporal_self_attention = IdentityOp
            return modified_submods

        # Replace any submodules that will have CP disabled and build them manually later after TransformerLayer init.
        modified_submods = _replace_no_cp_submodules(submodules)
        super().__init__(
            config=config, submodules=modified_submods, layer_number=layer_number, hidden_dropout=hidden_dropout
        )

        # Override Cross Attention to disable CP.
        # Disable TP Comm overlap as well. Not disabling will attempt re-use of buffer size same as
        #   Q and lead to incorrect tensor shapes.
        if submodules.cross_attention != IdentityOp:
            cp_override_config = copy.deepcopy(config)
            cp_override_config.context_parallel_size = 1
            cp_override_config.tp_comm_overlap = False
            self.cross_attention = build_module(
                submodules.cross_attention,
                config=cp_override_config,
                layer_number=layer_number,
            )
        else:
            self.cross_attention = None

        self.full_self_attention = build_module(
            submodules.full_self_attention,
            config=self.config,
            layer_number=layer_number,
        )

        self.adaLN = AdaLN(config=self.config, n_adaln_chunks=9 if self.cross_attention else 6)

    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
    ):

        rope_emb = rotary_pos_emb
        extra_pos_emb = packed_seq_params['extra_pos_emb']
        adaln_lora_B_3D = packed_seq_params['adaln_lora_B_3D']
        # if type(rotary_pos_emb) == list:
        #     rope_emb, extra_pos_emb = rotary_pos_emb
        #     # extra_pos_emb = rearrange(extra_pos_emb, 'B T H W D -> (T H W) B D')
        # else:
        #     rope_emb, extra_pos_emb = rotary_pos_emb, None

        if extra_pos_emb is not None:
            hidden_states = hidden_states + extra_pos_emb

        timestep_emb = attention_mask

        # ******************************************** full self attention *******************************************
        shift_ca, scale_ca, gate_ca = None, None, None
        if self.cross_attention:
            shift_full, scale_full, gate_full, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN(timestep_emb, adaln_lora_B_3D)
            )
        else:
            shift_full, scale_full, gate_full, shift_mlp, scale_mlp, gate_mlp = self.adaLN(
                timestep_emb, adaln_lora_B_3D
            )

        # print(f"megatron shift={shift_full}, scale={scale_full}, gate={gate_full}")
        # adaLN with scale + shift
        # print(f"megatron pre fa = {hidden_states}")
        pre_full_attn_layernorm_output_ada = self.adaLN.modulated_layernorm(
            hidden_states,
            shift=shift_full,
            scale=scale_full,
        )
        # z = rearrange(pre_full_attn_layernorm_output_ada, '(T H W) B D -> B T H W D', T=4, H=16, W=16)
        # print(f'megatron after adaLN: {z}, shape={z.shape}')
        attention_output, _ = self.full_self_attention(
            pre_full_attn_layernorm_output_ada,
            attention_mask=None,
            rotary_pos_emb=rope_emb,
            # packed_seq_params=packed_seq_params['self_attention'],
        )

        # print(f'megatron fa out={attention_output}, attention_output.shape={attention_output.shape}')

        if self.cross_attention:
            # ******************************************** cross attention ******************************************************
            # adaLN with scale + shift
            hidden_states, pre_cross_attn_layernorm_output_ada = self.adaLN.scaled_modulated_layernorm(
                residual=hidden_states,
                x=attention_output,
                gate=gate_full,
                shift=shift_ca,
                scale=scale_ca,
            )

            # print(f"megatron fa out={hidden_states}")

            attention_output, _ = self.cross_attention(
                pre_cross_attn_layernorm_output_ada,
                attention_mask=context_mask,
                key_value_states=context,
                # packed_seq_params=packed_seq_params['cross_attention'],
            )

        # print(f"mcore_dit_ca_out={attention_output}")

        # ******************************************** mlp ******************************************************
        hidden_states, pre_mlp_layernorm_output_ada = self.adaLN.scaled_modulated_layernorm(
            residual=hidden_states,
            x=attention_output,
            gate=gate_ca if self.cross_attention else gate_full,
            shift=shift_mlp,
            scale=scale_mlp,
        )

        # print(f"megatron pre mlp={hidden_states}")

        mlp_output, _ = self.mlp(pre_mlp_layernorm_output_ada)
        hidden_states = self.adaLN.scale_add(residual=hidden_states, x=mlp_output, gate=gate_mlp)

        # print(f"megatron mlp out={hidden_states}")

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)
        # output = hidden_states

        return output, context


import transformer_engine as te


def get_dit_adaln_block_with_transformer_engine_spec() -> ModuleSpec:
    """T5 decoder TE spec (uses Transformer Engine components)."""
    # from megatron.training import get_args
    # args = get_args()
    # params = {"attn_mask_type": AttnMaskType.padding if
    #            args.packing_algorithm != 'no_packing' else AttnMaskType.no_mask}
    params = {"attn_mask_type": AttnMaskType.no_mask}
    return ModuleSpec(
        module=DiTLayerWithAdaLN,
        submodules=DiTWithAdaLNSubmodules(
            full_self_attention=ModuleSpec(
                module=SelfAttention,
                params=params,
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=te.pytorch.RMSNorm,
                    k_layernorm=te.pytorch.RMSNorm,
                ),
            ),
            cross_attention=ModuleSpec(
                module=CrossAttention,
                params=params,
                submodules=CrossAttentionSubmodules(
                    linear_q=TEColumnParallelLinear,
                    linear_kv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=te.pytorch.RMSNorm,
                    k_layernorm=te.pytorch.RMSNorm,
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
        ),
    )
