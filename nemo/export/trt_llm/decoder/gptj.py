# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The GPTJ decoder implementation."""

from typing import Optional

from tensorrt_llm.models.gptj.model import GPTJDecoderLayer
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


class GPTJDecoderLayerConfigBuilder(DecoderLayerConfigBuilder):
    """The GPTJ implementation of the DecoderLayerConfigBuilder."""

    @override
    def hidden_act_fn(self, layer):
        """Returns the hidden act fn in the MLP layer, e.g. SiLUActivation or NewGELUActivation."""
        return layer.mlp.act

    @override
    def infer_num_attention_heads(self, layer):
        return layer.attn.num_attention_heads

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
            [layer.attn.q_proj, layer.attn.k_proj, layer.attn.v_proj],
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )

        config.dense = LinearConfig.from_nn_module(
            layer.attn.out_proj,
            LINEAR_ROW,
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )

        config.rotary_dim = layer.attn.rotary_dim

        return config

    @override
    def build_mlp(self, layer) -> MLPConfig:
        config = MLPConfig()
        config.fc = LinearConfig.from_nn_module(
            layer.mlp.fc_in,
            LINEAR_COLUMN,
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )
        config.proj = LinearConfig.from_nn_module(
            layer.mlp.fc_out,
            LINEAR_ROW,
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )

        return config

    @override
    def build_post_layernorm(self, layer) -> Optional[LayernormConfig]:
        # GPTJ do not have post layer_norm
        return None


class GPTJDecoderLayerBuilder(DecoderLayerBuilder):
    """The GPTJ implementation of the DecoderLayer."""

    @override
    def build_decoder(self, layer):
        assert self.tensor_parallel == 1 and self.rank == 0, "Only single GPU is supported for GPTJ"

        return GPTJDecoderLayer(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            rotary_dim=layer.attention.rotary_dim,
            dtype=self.dtype,
            hidden_act=self.hidden_act,
            tp_group=self.tp_group,
            tp_size=self.tensor_parallel,
        )
