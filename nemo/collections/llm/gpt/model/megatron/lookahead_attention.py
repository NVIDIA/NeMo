# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List, Optional

import torch
from nemo.utils import logging

try:
    from megatron.core import parallel_state
    from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
    from megatron.core.utils import make_viewless_tensor

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError) as e:

    HAVE_MEGATRON_CORE = False

    logging.warning("Megatron installation is required to use Look-Ahead Attention.")

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""


class LookAheadAttentionTransformerLayer(TransformerLayer):
    """A single transformer layer for Look-Ahead Attention.
    This technique aims modifies the transformer layer to maximize the overlap of FFNs and attention layers.
    It aims to increase e2e FMA utilization, and reduce latency in a deployed inference setting.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        lookahead_parallel_layers: Optional[List[int]] = None,
    ):
        """
        Args:
            lookahead_parallel_layers (List, optional): Pass in a list of two integers [A, B] to enable look-ahead
                attention. If enabled, layers A to B-1 (inclusive) use lookahead. A-1 is the transition layer
        """
        if not HAVE_MEGATRON_CORE:
            raise ImportError("megatron-core is required to use Look Ahead Attention.")
        super().__init__(
            config=config, submodules=submodules, layer_number=layer_number, hidden_dropout=hidden_dropout
        )

        if len(lookahead_parallel_layers) == 0:
            self.parallel_in = False
            self.parallel_out = False
        else:
            assert (
                config.pipeline_model_parallel_size == 1
            ), "LookAheadAttention does not support pipeline parallelism."
            assert len(lookahead_parallel_layers) == 2
            l = list(lookahead_parallel_layers)
            assert l[0] > 0 and l[1] < 61
            self.parallel_in = self.layer_number in range(l[0], l[1] + 1)
            self.parallel_out = self.layer_number in range(l[0] - 1, l[1])

        logging.info(f"{self.layer_number=}, {self.parallel_in=}, {self.parallel_out=}")

    def forward(
        self,
        hidden_states,
        attention_mask=None,
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
        """
        Perform a forward pass through the transformer layer.

        This method implements the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] or [2, s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask tensor for cross-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor, optional): Bias tensor for Q * K.T.
            inference_params (object, optional): Parameters for inference-time optimizations.
            packed_seq_params (object, optional): Parameters for packed sequence processing.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                output (Tensor): Transformed hidden states of shape [s, b, h].
                context (Tensor): Updated context tensor if cross-attention is used,
                otherwise None.
        """

        # Residual connection.
        if self.parallel_in:
            assert hidden_states.ndim == 4 and hidden_states.shape[0] == 2
            hidden_states, residual = torch.split(hidden_states, 1, dim=0)
            hidden_states = hidden_states.squeeze(dim=0)
            residual = residual.squeeze(dim=0)
        else:
            residual = hidden_states

        # =========================================================
        # ========== Code below same as TransformerLayer ==========

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_params=inference_params,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)

        # ========== Code above same as TransformerLayer ==========
        # =========================================================

        if self.parallel_out:
            assert residual.shape == output.shape
            output = torch.stack((residual, output), dim=0)

        # CUDA graph requires returned values to be Tensors
        if self.config.external_cuda_graph and self.training:
            return output

        return output, context
