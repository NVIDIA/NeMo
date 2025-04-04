# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from typing import Union

import torch
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
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
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import make_viewless_tensor
from megatron.core.transformer.attention import SelfAttention, CrossAttention
from megatron.core.transformer.attention import SelfAttentionSubmodules, CrossAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.extensions.transformer_engine import (
    TENorm,
    TERowParallelLinear,
    TEDotProductAttention,
    TEColumnParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

@dataclass
class PerceiverConfig(TransformerConfig):
    """Configuration class for Perceiver architecture.

    Extends TransformerConfig with Perceiver-specific parameters.
    
    Attributes:
        num_latents (int): Number of latent vectors in the latent array.
        num_self_attention_per_cross_attention (int): Number of self-attention heads per cross-attention head.
    """

    num_latents: int = 1
    num_self_attention_per_cross_attention: int = 1

@dataclass
class PerceiverLayerSubmodules:
    """Submodule specifications for components in a Perceiver layer.

    Contains specifications for both cross-attention and self-attention pathways.

    Attributes:
        cross_attention (Union[ModuleSpec, type]): Cross-attention module specification for processing input sequences.
        cross_attn_bda (Union[ModuleSpec, type]): Bias-dropout-add fusion for cross-attention outputs.
        self_attention (Union[ModuleSpec, type]): Self-attention module specification for processing latent states.
        self_attn_bda (Union[ModuleSpec, type]): Bias-dropout-add fusion for self-attention outputs.
        final_layernorm (Union[ModuleSpec, type]): Layer normalization applied to final latent states.
    """
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp
    
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    final_layernorm: Union[ModuleSpec, type] = IdentityOp

class PerceiverLayer(MegatronModule):
    """A single layer in the Perceiver architecture.

    The Perceiver layer processes input sequences through cross-attention followed by
    multiple self-attention layers on a latent array. This allows the model to handle 
    arbitrary input sequences by projecting them into a fixed-size latent space.

    Args:
        config (PerceiverConfig): Configuration object containing model parameters.
        submodules (PerceiverLayerSubmodules): Submodule specifications for layer components.
        layer_number (int): Index of this layer in the stack. Defaults to 1.

    Attributes:
        input_cross_attention (CrossAttention): Cross-attention module from latents to input sequence.
        input_cross_attn_bda (callable): Bias-dropout-add fusion for cross-attention.
        self_attention_layers (torch.nn.ModuleList): List of self-attention modules operating on latents.
        final_layernorm (TENorm): Final layer normalization on latent states.
    """

    def __init__(
        self,
        config: PerceiverConfig,
        submodules: PerceiverLayerSubmodules,
        layer_number: int = 1,
    ):
        super().__init__(config=config)

        self.config = config

        self.submodules_config = submodules
        self.layer_number = layer_number
        
        self.input_cross_attention = build_module(
            submodules.cross_attention, config=self.config, layer_number=layer_number
        )

        self.input_cross_attn_bda = build_module(submodules.cross_attn_bda)

        self.self_attention_layers = torch.nn.ModuleList([
            build_module(
                submodules.self_attention,
                config=self.config,
                layer_number=layer_number
            ) for _ in range(self.config.num_self_attention_per_cross_attention)
        ])

        # Final LayerNorm on the latent array
        self.final_layernorm = build_module(
            submodules.final_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

    def forward(
        self,
        latent_states: torch.Tensor,
        input_sequence: torch.Tensor,
        cross_attention_masks: torch.Tensor = None,
        self_attention_mask: torch.Tensor = None,
        inference_params = None,
    ):
        """Forward pass through a Perceiver layer.

        Args:
            latent_states (torch.Tensor): Latent array of shape [latent_len, batch_size, hidden_size]
                that learns to attend to the input sequence.
            input_sequence (torch.Tensor): Input sequence of shape [seq_len, batch_size, hidden_size]
                containing the actual input data.
            cross_attention_masks (torch.Tensor): Tuple of attention masks (query_mask, key_value_mask) for
                cross-attention. Each mask has shape [batch_size, 1, 1, seq_len].
            self_attention_mask (torch.Tensor): Attention mask for self-attention on latents,
                with shape [batch_size, 1, 1, latent_len].
            inference_params (Optional[dict]): Optional parameters to optimize inference-time computation.

        Returns:
            torch.Tensor: Updated latent states after processing through the layer,
                with shape [latent_len, batch_size, hidden_size].
        """
        print("\nShape tracking through forward pass:")
        print(f"Initial latent_states: {latent_states.shape}")  # [latent_len, batch_size, hidden_size]
        print(f"Input sequence: {input_sequence.shape}")  # [seq_len, batch_size, hidden_size]

        # Cross attention from latents to input sequence
        cross_attn_output = self.input_cross_attention(
            latent_states,
            key_value_states=input_sequence,
            attention_mask=cross_attention_masks,
            inference_params=inference_params,
        )
        print(f"After cross attention - output: {cross_attn_output[0].shape}, bias: {cross_attn_output[1].shape}")
        latent_states = cross_attn_output

        # Multiple self-attention layers
        for i, self_attn_layer in enumerate(self.self_attention_layers):
            self_attn_output = self_attn_layer(
                latent_states[0],
                attention_mask=self_attention_mask,
                inference_params=inference_params,
            )
            print(f"After self attention layer {i} - output: {self_attn_output[0].shape}, bias: {self_attn_output[1].shape}")
            latent_states = self_attn_output

        latent_states = self.final_layernorm(latent_states[0])
        print(f"After final layernorm: {latent_states.shape}")

        output = make_viewless_tensor(
            inp=latent_states,
            requires_grad=latent_states.requires_grad,
            keep_graph=True
        )
        print(f"Final output: {output.shape}")

        return output


# Example ModuleSpec for Perceiver layer
perceiver_layer_spec = ModuleSpec(
    module=PerceiverLayer,
    submodules=PerceiverLayerSubmodules(
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
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.padding},
            submodules=SelfAttentionSubmodules(
                linear_qkv=TEColumnParallelLinear,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        final_layernorm=ModuleSpec(
            module=TENorm
        ),
    )
)
