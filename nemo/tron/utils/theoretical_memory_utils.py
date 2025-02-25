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

"""Computes theoretical memory footprint for model training."""

import math

from nemo.tron.config import ConfigContainer

NUM_BYTES_IN_MEGABYTE = 1024 * 1024


def compute_weight_and_optimizer_memory(config: ConfigContainer, verbose=False):
    model_config = config.model_config
    mlm_config = config.megatron_lm_config
    # Attention projection size.
    query_projection_size = model_config.kv_channels * model_config.num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / model_config.hidden_size
    # Group Query Attention.
    num_query_groups = (
        model_config.num_query_groups if model_config.num_query_groups else model_config.num_attention_heads
    )
    # MoE.
    num_experts = 1 if model_config.num_moe_experts is None else model_config.num_moe_experts
    gated_linear_multiplier = 3 / 2 if mlm_config.swiglu else 1
    num_parameters_in_transformer_layers = (
        2
        * model_config.num_layers
        * model_config.hidden_size
        * model_config.hidden_size
        * (
            # Attention.
            ((1 + (num_query_groups / model_config.num_attention_heads)) * query_projection_to_hidden_size_ratio)
            # MLP.
            + ((model_config.ffn_hidden_size / model_config.hidden_size) * num_experts * gated_linear_multiplier)
            # Transformer layernorms.
            + (2 / model_config.hidden_size)
            # Final layernorm.
            + (1 / (model_config.num_layers * model_config.hidden_size))
        )
    )
    embedding_size = model_config.hidden_size * config.tokenizer_config.padded_vocab_size
    if mlm_config.untie_embeddings_and_output_weights:
        num_parameters_in_embedding_layers = 2 * embedding_size
    else:
        num_parameters_in_embedding_layers = embedding_size
    num_total_parameters = num_parameters_in_transformer_layers + num_parameters_in_embedding_layers
    if verbose:
        print(
            f"Number of parameters in transformer layers in billions: "
            f"{num_parameters_in_transformer_layers / 10**9: .2f}"
        )
        print(
            f"Number of parameters in embedding layers in billions: {num_parameters_in_embedding_layers / 10**9:.2f}"
        )
        print(f"Total number of parameters in billions: {num_total_parameters / 10**9:.2f}")

    # Most loaded model shard has (1/pp_size transformer layers + 1 embedding layer) / tp_size.
    num_parameters_on_most_loaded_model_shard = (
        (num_parameters_in_transformer_layers / model_config.pipeline_model_parallel_size) + embedding_size
    ) / model_config.tensor_model_parallel_size
    if mlm_config.untie_embeddings_and_output_weights and model_config.pipeline_model_parallel_size == 1:
        num_parameters_on_most_loaded_model_shard += embedding_size / model_config.tensor_model_parallel_size
    if verbose:
        print(
            f"Number of parameters in most loaded shard in billions: "
            f"{num_parameters_on_most_loaded_model_shard / 10**9:.4f}"
        )

    if model_config.pipeline_model_parallel_size > 1:
        # Other shards just have (1/pp_size transformer layers) / tp_size.
        num_parameters_on_other_model_shards = num_parameters_in_transformer_layers / (
            model_config.pipeline_model_parallel_size * model_config.tensor_model_parallel_size
        )
        if verbose:
            print(
                f"Number of parameters in other shards in billions: {num_parameters_on_other_model_shards / 10**9:.4f}"
            )

    num_bytes_per_parameter = (
        18 if not config.optimizer_config.use_distributed_optimizer else 6 + (12 / config.data_parallel_size)
    )
    weight_and_optimizer_memory = num_parameters_on_most_loaded_model_shard * num_bytes_per_parameter

    return weight_and_optimizer_memory


def compute_activation_memory(config: ConfigContainer, num_microbatches, verbose=False):
    # Using formula in Table 2 of https://arxiv.org/pdf/2205.05198.pdf.
    # We are trying to compute the maximum activation footprint, so all calculations in this
    # function are for the first pipeline stage.

    # TODO: This function needs to take into account query_projection_size potentially being
    # different from hidden_size.

    model_config = config.model_config
    mlm_config = config.megatron_lm_config

    # Memory footprint from transformer layer (self-attention and MLP).
    activation_memory = (model_config.seq_length * mlm_config.micro_batch_size * model_config.hidden_size) * (
        18 + (4 * (model_config.ffn_hidden_size / model_config.hidden_size))
    )
    if verbose:
        print(
            f"Activation memory footprint per transformer layer: "
            f"{activation_memory / NUM_BYTES_IN_MEGABYTE / model_config.tensor_model_parallel_size:.1f} MB"
        )
    activation_memory *= model_config.num_layers

    # Now add activation memory required for input embeddings, last LayerNorm and output layer.

    # Input to embedding (pp_size microbatches in flight).
    activation_memory += (
        8 * model_config.seq_length * mlm_config.micro_batch_size * model_config.pipeline_model_parallel_size
    )
    # Dropout in embedding layer (pp_size microbatches in flight).
    activation_memory += (
        model_config.seq_length
        * mlm_config.micro_batch_size
        * model_config.hidden_size
        * model_config.pipeline_model_parallel_size
    )

    # Multiply by interleaved PP memory factor.
    if model_config.virtual_pipeline_model_parallel_size is not None:
        interleaved_schedule_memory_penalty = 1 + (
            (model_config.pipeline_model_parallel_size - 1)
            / (model_config.pipeline_model_parallel_size * model_config.virtual_pipeline_model_parallel_size)
        )
        in_flight_microbatches = math.ceil(
            interleaved_schedule_memory_penalty * model_config.pipeline_model_parallel_size
        )
        if verbose:
            print(f"Memory penalty from interleaved schedule: {interleaved_schedule_memory_penalty:.2f}")
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")
        activation_memory *= interleaved_schedule_memory_penalty

    # If using non-interleaved schedule, number of microbatches in pipeline can be less than pp_size,
    # so discount accordingly.
    if model_config.virtual_pipeline_model_parallel_size is None and model_config.pipeline_model_parallel_size > 1:
        if num_microbatches is not None:
            activation_memory *= min(1, num_microbatches / model_config.pipeline_model_parallel_size)
            in_flight_microbatches = min(num_microbatches, model_config.pipeline_model_parallel_size)
        else:
            in_flight_microbatches = model_config.pipeline_model_parallel_size
        if verbose:
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")

    if model_config.pipeline_model_parallel_size == 1:
        # Inputs to output layer and CE loss.
        activation_memory += (
            model_config.seq_length
            * mlm_config.micro_batch_size
            * model_config.hidden_size
            * 4
            * (1 + (config.tokenizer_config.padded_vocab_size / model_config.hidden_size))
        )

    # Activation memory is partitioned by TP size due to tensor and sequence model parallelism.
    return activation_memory / model_config.tensor_model_parallel_size


def report_theoretical_memory(config: ConfigContainer, num_microbatches=None, verbose=False):
    weight_and_optimizer_memory = compute_weight_and_optimizer_memory(config, verbose=verbose) / NUM_BYTES_IN_MEGABYTE

    # Formulae here assume sequence parallelism and selective activation recomputation.
    if not config.model_config.sequence_parallel or config.model_config.recompute_granularity != "selective":
        print(f"Theoretical memory footprints: weight and optimizer={weight_and_optimizer_memory:.2f} MB")
        return

    activation_memory = (
        compute_activation_memory(config, num_microbatches=num_microbatches, verbose=verbose) / NUM_BYTES_IN_MEGABYTE
    )
    total_memory = weight_and_optimizer_memory + activation_memory

    print(
        f"Theoretical memory footprints: weight and optimizer={weight_and_optimizer_memory:.2f} MB, "
        f"activation={activation_memory:.2f} MB, total={total_memory:.2f} MB\n"
    )
