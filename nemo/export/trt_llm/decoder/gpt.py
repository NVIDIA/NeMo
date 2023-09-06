# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The GPT2 decoder implementation."""

from typing import Optional

from tensorrt_llm.layers import AttentionMaskType, PositionEmbeddingType
from tensorrt_llm.models.gpt.model import GPTDecoderLayer
from typing_extensions import override

from ..model_config import (
    LINEAR_COLUMN,
    LINEAR_ROW,
    AttentionConfig,
    LayernormConfig,
    LinearConfig,
    MLPConfig,
)
from .decoder import DecoderLayerBuilder, DecoderLayerConfigBuilder


class GPTDecoderLayerConfigBuilder(DecoderLayerConfigBuilder):
    """The GPT2 implementation of the DecoderLayerConfigBuilder."""

    @override
    def hidden_act_fn(self, layer):
        return layer.mlp.act

    @override
    def infer_num_attention_heads(self, layer):
        return layer.attn.num_heads

    @override
    def infer_max_position_embeddings(self, layer):
        return layer.attn.bias.shape[2]

    @override
    def build_input_layernorm(self, layer) -> LayernormConfig:
        return LayernormConfig.from_nn_module(layer.ln_1, dtype=self.dtype)

    @override
    def build_attention(self, layer) -> AttentionConfig:
        config = AttentionConfig()
        config.qkv = LinearConfig.from_qkv_nn_modules(
            [layer.attn.c_attn],
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )

        config.dense = LinearConfig.from_nn_module(
            layer.attn.c_proj,
            LINEAR_ROW,
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )

        return config

    @override
    def build_mlp(self, layer) -> MLPConfig:
        config = MLPConfig()
        config.fc = LinearConfig.from_nn_module(
            layer.mlp.c_fc,
            LINEAR_COLUMN,
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )
        config.proj = LinearConfig.from_nn_module(
            layer.mlp.c_proj,
            LINEAR_ROW,
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )

        return config

    @override
    def build_post_layernorm(self, layer) -> Optional[LayernormConfig]:
        return LayernormConfig.from_nn_module(layer.ln_2, dtype=self.dtype)


class GPTDecoderLayerBuilder(DecoderLayerBuilder):
    """The GPT implementation of the DecoderLayer."""

    @override
    def build_decoder(self, layer):
        rotary_pct = layer.rotary_pct
        position_embedding_type = (
            PositionEmbeddingType.learned_absolute
            if rotary_pct == 0.0
            else PositionEmbeddingType.rope_gpt_neox
        )

        bias_qkv = layer.attention.qkv.bias is not None

        return GPTDecoderLayer(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            num_layers=self.num_layers,
            dtype=self.dtype,
            apply_query_key_layer_scaling=False,
            attention_mask_type=AttentionMaskType.causal,
            hidden_act=self.hidden_act,
            position_embedding_type=position_embedding_type,
            rotary_embedding_percentage=rotary_pct,
            inter_size=layer.ffn_hidden_size_local * self.tensor_parallel,
            bias=bias_qkv,
            tp_group=self.tp_group,
            tp_size=self.tensor_parallel,
        )
