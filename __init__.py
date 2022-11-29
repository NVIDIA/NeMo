# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""Model parallel utility interface."""

from apex.transformer.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy

from apex.transformer.tensor_parallel.data import broadcast_data

from apex.transformer.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    set_tensor_model_parallel_attributes,
    set_defaults_if_not_set_tensor_model_parallel_attributes,
    copy_tensor_model_parallel_attributes,
)

from apex.transformer.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)

from .random import (
    checkpoint,
    get_cuda_rng_tracker,
    init_checkpointed_activations_memory_buffer,
    model_parallel_cuda_manual_seed,
    reset_checkpointed_activations_memory_buffer,
)

from apex.transformer.tensor_parallel.utils import split_tensor_along_last_dim


__all__ = [
    # cross_entropy.py
    "vocab_parallel_cross_entropy",
    # data.py
    "broadcast_data",
    # layers.py
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "set_tensor_model_parallel_attributes",
    "set_defaults_if_not_set_tensor_model_parallel_attributes",
    "copy_tensor_model_parallel_attributes",
    # mappings.py
    "copy_to_tensor_model_parallel_region",
    "gather_from_tensor_model_parallel_region",
    "reduce_from_tensor_model_parallel_region",
    "scatter_to_tensor_model_parallel_region",
    "scatter_to_sequence_parallel_region",
    # random.py
    "checkpoint",
    "get_cuda_rng_tracker",
    "init_checkpointed_activations_memory_buffer",
    "model_parallel_cuda_manual_seed",
    "reset_checkpointed_activations_memory_buffer",
    # utils.py
    "split_tensor_along_last_dim",
]
