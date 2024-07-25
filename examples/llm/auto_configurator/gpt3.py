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


# GPT-3 126M default model
# Create auto configurator runner
runner = AutoConfigurator(
    model_type="gpt3",
    model_size=126,
    model_measure="M",
    num_nodes=8,
    data_paths=["/path/to/datafile1", "/path/to/datafile_2"],
)

# Get generated configs
configs = runner.get_configs()


# GPT-3 5B default model
# Create auto configurator runner
runner = AutoConfigurator(
    model_type="gpt3",
    model_size=5,
    global_batch_size=2048,
    seq_length=2048,
    num_nodes=16,
    data_paths=["/path/to/datafile1", "/path/to/datafile_2"],
)

# Get generated configs
configs = runner.get_configs()


# GPT-3 uknown size with additional args
# Create auto configurator runner
runner = AutoConfigurator(
    model_type="gpt3",
    model_size=None,
    num_nodes=64,
    global_batch_size=2048,
    seq_length=2048,
    tensor_parallel_sizes=[2, 4, 8],
    pipeline_parallel_sizes=[1, 2, 4],
    micro_batch_sizes=[1, 2],
    min_model_parallel_size=2,
    max_model_parallel_size=16,
    max_minutes_per_run=30,
    max_steps_per_run=50,
    max_training_days=14,
    data_paths=["/path/to/datafile1", "/path/to/datafile_2"],
    model_args={"sequence_parallel": True, "layernorm_epsilon": 1e-4, "apply_query_key_layer_scaling": True},
)

# Get generated configs
configs = runner.get_configs()
