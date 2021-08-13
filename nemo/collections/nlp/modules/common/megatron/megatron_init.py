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

from megatron.initialize import initialize_megatron
from megatron.mpu.initialize import set_pipeline_model_parallel_rank, set_pipeline_model_parallel_world_size


def initialize_megatron_for_nemo(
    micro_batch_size=1,
    tensor_model_parallel_size=1,
    encoder_seq_length=512,
    num_layers=1,
    hidden_size=16,
    num_attention_heads=1,
    max_position_embeddings=512,
):
    args_defaults = {}
    args_defaults['num_layers'] = num_layers
    args_defaults['hidden_size'] = hidden_size
    args_defaults['num_attention_heads'] = num_attention_heads
    args_defaults['max_position_embeddings'] = max_position_embeddings

    extra_args_provider = get_extra_args_provider(micro_batch_size, tensor_model_parallel_size, encoder_seq_length)

    # pipeline model parallelism not implemented in NeMo yet
    set_pipeline_model_parallel_rank(0)
    set_pipeline_model_parallel_world_size(1)

    initialize_megatron(extra_args_provider=extra_args_provider, args_defaults=args_defaults)


def get_extra_args_provider(
    micro_batch_size=1, tensor_model_parallel_size=1, encoder_seq_length=512,
):
    def extra_args_provider(parser):
        parser.set_defaults(micro_batch_size=micro_batch_size)
        parser.set_defaults(tensor_model_parallel_size=tensor_model_parallel_size)
        parser.set_defaults(encoder_seq_length=encoder_seq_length)
        return parser

    return extra_args_provider
