# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.llm.tools.auto_configurator import AutoConfigurator


# Llama2 7B default model
# Create auto configurator runner
runner = AutoConfigurator(
    model_type="llama",
    model_size=7,
    model_version=2,
    global_batch_size=2048,
    seq_length=4096,
    num_nodes=16,
    tokenizer_path=["/path/to/tokenizer.model"],
    data_paths=["/path/to/datafile1", "/path/to/datafile_2"],
)

# Get generated configs
configs = runner.get_configs()


# Llama3 70B default model
# Create auto configurator runner
runner = AutoConfigurator(
    model_type="llama",
    model_size=70,
    model_version=3,
    num_nodes=128,
    global_batch_size=2048,
    seq_length=8192,
    tensor_parallel_sizes=[4, 8],
    pipeline_parallel_sizes=[2, 4],
    micro_batch_sizes=[1],
    context_parallel_sizes=[1, 2],
    min_model_parallel_size=8,
    max_model_parallel_size=32,
    max_minutes_per_run=30,
    max_steps_per_run=25,
    tokenizer_path=["/path/to/tokenizer.model"],
    data_paths=["/path/to/datafile1", "/path/to/datafile_2"],
)

# Get generated configs
configs = runner.get_configs()


# Llama uknown model size with additional args
# Create auto configurator runner
runner = AutoConfigurator(
    model_type="llama",
    num_nodes=32,
    global_batch_size=2048,
    seq_length=8192,
    context_parallel_sizes=[1, 2],
    min_model_parallel_size=2,
    max_model_parallel_size=8,
    max_minutes_per_run=20,
    max_training_days=28,
    tflops_per_gpu=100,
    num_tokens_in_b=600,
    max_steps_per_run=25,
    tokenizer_path=["/path/to/tokenizer.model"],
    data_paths=["/path/to/datafile1", "/path/to/datafile_2"],
    model_args={"layernorm_epsilon": 1e-4, "apply_query_key_layer_scaling": True},
)

# Get generated configs
configs = runner.get_configs()


