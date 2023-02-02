# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch.nn as nn

from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil
from nemo.collections.common.parts.utils import activation_registry
from nemo.collections.nlp.modules.common.megatron.attention import ParallelAttention_
from nemo.collections.nlp.modules.common.megatron.utils import init_method_const, init_method_normal
from nemo.core.classes.mixins import adapter_mixin_strategies

try:
    from apex.transformer.tensor_parallel import RowParallelLinear, ColumnParallelLinear
    from apex.transformer.enums import AttnMaskType, AttnType
    from apex.normalization.fused_layer_norm import MixedFusedLayerNorm

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False


class ParallelLinearAdapter(nn.Module, AdapterModuleUtil):
    def __init__(
        self,
        in_features: int,
        dim: int,
        activation: str = 'swish',
        norm_position: str = 'post',
        norm_type: str = 'mixedfusedlayernorm',
        column_init_method: str = 'xavier',
        row_init_method: str = 'zero',
        dropout: float = 0.0,
        adapter_strategy: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig = None,
    ):
        super().__init__()
        if not HAVE_APEX:
            logging.info("Apex is required to use ParallelLinearAdapters.")
            raise RuntimeError("ParallelLinearAdapter can not run without Apex.")
        self.activation = activation_registry[activation]()
        self.norm_position = norm_position

        if column_init_method == 'xavier':
            self.linear_in = ColumnParallelLinear(in_features, dim, bias=False)
        elif column_init_method == 'normal':
            self.linear_in = ColumnParallelLinear(in_features, dim, bias=False, init_method=init_method_normal(0.2))
        elif column_init_method == 'zero':
            self.linear_in = ColumnParallelLinear(in_features, dim, bias=False, init_method=init_method_const(0.0))
        else:
            raise NotImplementedError("column_init_method should be zero, normal or xavier")

        if row_init_method == 'xavier':
            self.linear_out = RowParallelLinear(dim, in_features, bias=False)
        elif row_init_method == 'normal':
            self.linear_out = RowParallelLinear(dim, in_features, bias=False, init_method=init_method_normal(0.2))
        elif row_init_method == 'zero':
            self.linear_out = RowParallelLinear(dim, in_features, bias=False, init_method=init_method_const(0.0))
        else:
            raise NotImplementedError("row_init_method should be zero, normal or xavier")

        if norm_type == 'mixedfusedlayernorm':
            self.layer_norm = MixedFusedLayerNorm(in_features, 1e-5, sequence_parallel_enbaled=False)
        elif norm_type == 'layernorm':
            self.layer_norm = nn.LayerNorm(in_features)
        else:
            raise NotImplementedError("norm_type should be either mixedfusedlayernorm or layernorm")

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

    def forward(self, x):

        if self.norm_position == 'pre':
            x = self.layer_norm(x)

        x, _ = self.linear_in(x)  # (@adithyare) ColumnLinear returns output and bias, we are ignoring the bias term.
        x = self.activation(x)
        x, _ = self.linear_out(x)

        if self.norm_position == 'post':
            x = self.layer_norm(x)

        # Add dropout if available
        if self.dropout is not None:
            x = self.dropout(x)

        return x


class TinyAttentionAdapter(nn.Module, AdapterModuleUtil):
    def __init__(
        self,
        norm_type: str = "mixedfusedlayernorm",
        init_method=init_method_normal(0.015),
        output_layer_init_method=init_method_normal(0.015),  # TODO (@adithyare) revisit init for output layer
        layer_number: int = 1,
        num_attention_heads: int = 1,
        hidden_size: int = 2048,
        attention_type=AttnType.self_attn,
        attn_mask_type=AttnMaskType.padding,
        precision=16,
        apply_query_key_layer_scaling=True,
        kv_channels: int = 16,
        use_cpu_initialization=False,
        masked_softmax_fusion=True,
        attention_dropout=0.1,
        layer_type=None,
        megatron_legacy=False,
        bias=True,
        headscale=False,
        activations_checkpoint_granularity=None,
        sequence_parallel=False,
        gradient_accumulation_fusion=False,
        normalize_attention_scores=True,
        adapter_strategy: Optional[Any] = adapter_mixin_strategies.ResidualAddAdapterStrategyConfig(),
        activation: str = 'swish',
    ):
        super().__init__()
        if not HAVE_APEX:
            logging.info("Apex is required to use TinyAttentionAdapter.")
            raise RuntimeError("TinyAttentionAdapter can not run without Apex.")
        self.activation = activation_registry[activation]()
        self.tiny_attention = ParallelAttention_(
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
            precision=precision,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            use_cpu_initialization=use_cpu_initialization,
            masked_softmax_fusion=masked_softmax_fusion,
            attention_dropout=attention_dropout,
            layer_type=layer_type,
            megatron_legacy=megatron_legacy,
            bias=bias,
            headscale=headscale,
            activations_checkpoint_granularity=activations_checkpoint_granularity,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            normalize_attention_scores=normalize_attention_scores,
        )
        if norm_type == 'mixedfusedlayernorm':
            self.layer_norm = MixedFusedLayerNorm(hidden_size, 1e-5, sequence_parallel_enbaled=False)
        elif norm_type == 'layernorm':
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            raise NotImplementedError("norm_type should be either mixedfusedlayernorm or layernorm")
        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_past=None,
        get_key_value=False,
        encoder_output=None,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        rotary_pos_emb=None,  # rotary positional embedding
        relative_position_bias=None,
        checkpoint_core_attention=False,
    ):
        hidden_states = self.layer_norm(hidden_states)
        attention_output, attention_bias = self.tiny_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            get_key_value=get_key_value,
            encoder_output=encoder_output,
            set_inference_key_value_memory=set_inference_key_value_memory,
            inference_max_sequence_len=inference_max_sequence_len,
            rotary_pos_emb=rotary_pos_emb,
            relative_position_bias=relative_position_bias,
            checkpoint_core_attention=checkpoint_core_attention,
        )
        return attention_output, attention_bias


@dataclass
class ParallelLinearAdapterConfig:
    in_features: int
    dim: int
    activation: str = 'swish'
    norm_position: str = 'post'
    norm_type: str = 'mixedfusedlayernorm'
    column_init_method: str = 'xavier'
    row_init_method: str = 'zero'
    dropout: float = 0.0
    adapter_strategy: Optional[Any] = adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()
    _target_: str = "{0}.{1}".format(ParallelLinearAdapter.__module__, ParallelLinearAdapter.__name__)


@dataclass
class TinyAttentionAdapterConfig:
    norm_type: str = "mixedfusedlayernorm"
    init_method = init_method_normal(0.015)
    output_layer_init_method = init_method_normal(0.015)  # TODO (@adithyare) revisit init for output layer
    layer_number: int = 1
    num_attention_heads: int = 1
    hidden_size: int = 2048
    attention_type = AttnType.self_attn
    attn_mask_type = AttnMaskType.padding
    precision = 16
    apply_query_key_layer_scaling = True
    kv_channels: int = 128
    use_cpu_initialization = False
    masked_softmax_fusion = True
    attention_dropout = 0.1
    layer_type = None
    megatron_legacy = False
    bias = True
    headscale = False
    activations_checkpoint_granularity = None
    sequence_parallel = False
    gradient_accumulation_fusion = False
    normalize_attention_scores = True
    adapter_strategy: Optional[Any] = adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()
    _target_: str = "{0}.{1}".format(TinyAttentionAdapter.__module__, TinyAttentionAdapter.__name__)
