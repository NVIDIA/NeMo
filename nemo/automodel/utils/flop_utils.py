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

from typing import Optional

from transformers import PretrainedConfig

from nemo.automodel.config import ConfigContainer


def num_floating_point_operations(cfg: ConfigContainer, model_config: Optional[PretrainedConfig], batch_size: int):
    if model_config is None:
        return 0.0

    # Attention projection size.
    if getattr(model_config, "head_dim", None):
        query_projection_size = model_config.head_dim * model_config.num_attention_heads
    else:
        query_projection_size = model_config.hidden_size

    query_projection_to_hidden_size_ratio = query_projection_size / model_config.hidden_size
    # Group Query Attention.
    if not model_config.num_key_value_heads:
        num_query_groups = model_config.num_attention_heads
    else:
        num_query_groups = model_config.num_key_value_heads

    # MoE.
    # num_experts_routed_to = 1 if model_config.num_moe_experts is None else model_config.moe_router_topk
    num_experts_routed_to = 1
    gated_linear_multiplier = 3 / 2 if model_config.hidden_act == "silu" else 1
    shared_expert_ffn_hidden_size = 0  # TODO: Change when adding MoE support

    # The 12x term below comes from the following factors; for more details, see
    # "APPENDIX: FLOATING-POINT OPERATIONS" in https://arxiv.org/abs/2104.04473.
    # - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
    #       backward wgrad [weight gradient], backward dgrad [data gradient]).
    # - 2x: GEMMs of a particular size are stacked twice in the standard Transformer model
    #       architectures implemented in this codebase (e.g., h->ffn_h GEMM and ffn_h->h GEMM
    #       in MLP layer).
    # - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
    expansion_factor = 3 * 2 * 2

    return (
        expansion_factor
        * batch_size
        * cfg.dataset_config.seq_length
        * model_config.num_hidden_layers
        * model_config.hidden_size
        * model_config.hidden_size
        * (
            # Attention.
            (
                (
                    1
                    + (num_query_groups / model_config.num_attention_heads)
                    + (cfg.dataset_config.seq_length / model_config.hidden_size)
                )
                * query_projection_to_hidden_size_ratio
            )
            # MLP.
            + (
                (model_config.intermediate_size / model_config.hidden_size)
                * num_experts_routed_to
                * gated_linear_multiplier
            )
            # Shared Experts.
            + ((shared_expert_ffn_hidden_size / model_config.hidden_size) * gated_linear_multiplier)
            # Logit.
            + (model_config.vocab_size / (2 * model_config.num_hidden_layers * model_config.hidden_size))
        )
    )
