# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass
from typing import Union

import torch
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import (
    CrossAttention,
    CrossAttentionSubmodules,
    SelfAttention,
    SelfAttentionSubmodules,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor


@dataclass
class PerceiverConfig(TransformerConfig):
    """Configuration class for Perceiver architecture.

    Extends TransformerConfig with Perceiver-specific parameters.

    Additional Args:
        num_latents: Number of latent vectors in the latent array
    """

    num_latents: int = 1


@dataclass
class PerceiverLayerSubmodules:
    """Perceiver layer submodules."""

    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    latent_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp


class PerceiverLayer(MegatronModule):
    """A single perceiver layer.

    Perceiver layer takes input with size [s, b, h] and latent array [l, b, h]
    where l is the (smaller) latent dimension, and returns processed latents
    of size [l, b, h].
    """

    def __init__(
        self,
        config: PerceiverConfig,
        submodules: PerceiverLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
    ):
        super().__init__(config=config)

        self.config = config

        self.submodules_config = submodules
        self.layer_number = layer_number
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        # First cross attention from latents to input sequence
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.input_cross_attention = build_module(
            submodules.cross_attention, config=self.config, layer_number=layer_number
        )

        self.input_cross_attn_bda = build_module(submodules.cross_attn_bda)

        # Self attention on latent array
        self.latent_layernorm = build_module(
            submodules.latent_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.self_attention = build_module(submodules.self_attention, config=self.config, layer_number=layer_number)

        self.self_attn_bda = build_module(submodules.self_attn_bda)

        # MLP block
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm, config=self.config, hidden_size=self.config.hidden_size
        )

        self.mlp = build_module(submodules.mlp, config=self.config)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        self.mlp_bda = build_module(submodules.mlp_bda)

        self.bias_dropout_add_exec_handler = torch.enable_grad

    def forward(
        self,
        latent_states: torch.Tensor,
        input_sequence: torch.Tensor,
        cross_attention_masks,
        self_attention_mask,
        inference_params=None,
    ):
        """
        Args:
            latent_states (Tensor): Latent array of shape [l, b, h]
            input_sequence (Tensor): Input sequence of shape [s, b, h]
            cross_attention_masks (tuple): Tuple of (cross_q, cross_kv)
                where each mask has shape [b, 1, 1, seq_len]
            self_attention_masks (tuple): Tuple of (self_q, self_kv)
                where each mask has shape [b, 1, 1, seq_len]
            inference_params: Parameters for inference optimizations
        """
        # Cross attention from latents to input sequence
        residual = latent_states
        norm_latents = self.input_layernorm(latent_states)

        cross_attn_output = self.input_cross_attention(
            norm_latents,
            key_value_states=input_sequence,
            attention_mask=cross_attention_masks,
            inference_params=inference_params,
        )

        with self.bias_dropout_add_exec_handler():
            latent_states = self.input_cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                cross_attn_output, residual, self.hidden_dropout
            )

        # Self attention on latent array
        residual = latent_states
        norm_latents = self.latent_layernorm(latent_states)

        self_attn_output = self.self_attention(
            norm_latents,
            attention_mask=self_attention_mask,
            inference_params=inference_params,
        )

        with self.bias_dropout_add_exec_handler():
            latent_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                self_attn_output, residual, self.hidden_dropout
            )

        # MLP
        residual = latent_states
        norm_latents = self.pre_mlp_layernorm(latent_states)
        mlp_output = self.mlp(norm_latents)

        with self.bias_dropout_add_exec_handler():
            latent_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output, residual, self.hidden_dropout
            )

        output = make_viewless_tensor(inp=latent_states, requires_grad=latent_states.requires_grad, keep_graph=True)

        return output


# Example ModuleSpec for Perceiver layer
perceiver_layer_spec = ModuleSpec(
    module=PerceiverLayer,
    submodules=PerceiverLayerSubmodules(
        input_layernorm=ModuleSpec(module=TENorm),
        cross_attention=ModuleSpec(
            module=CrossAttention,
            params={"attn_mask_type": AttnMaskType.padding},
            submodules=CrossAttentionSubmodules(
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
                linear_q=TEColumnParallelLinear,
                linear_kv=TEColumnParallelLinear,
            ),
        ),
        cross_attn_bda=get_bias_dropout_add,
        latent_layernorm=ModuleSpec(module=TENorm),
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.padding},
            submodules=SelfAttentionSubmodules(
                linear_qkv=TELayerNormColumnParallelLinear,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=ModuleSpec(module=TENorm),
        mlp=ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ),
        ),
        mlp_bda=get_bias_dropout_add,
    ),
)
