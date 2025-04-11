# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
export CUDA_VISIBLE_DEVICES=0
mkdir examples/llm/auto_configurator/auto_conf_logs_llama

export CUDA_VISIBLE_DEVICES=0,1
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo -m torch.distributed.launch --nproc_per_node=2 --use-env examples/llm/auto_configurator/auto_config.py \
    --model_type=llama \
    --log_dir=/workspace/examples/llm/auto_configurator/auto_conf_logs_llama \
    --run_number=1

export CUDA_VISIBLE_DEVICES=0,1
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo -m torch.distributed.launch --nproc_per_node=2 --use-env examples/llm/auto_configurator/auto_config.py \
    --model_type=llama \
    --log_dir=/workspace/examples/llm/auto_configurator/auto_conf_logs_llama \
    --run_number=2

export CUDA_VISIBLE_DEVICES=0,1
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo -m torch.distributed.launch --nproc_per_node=2 --use-env examples/llm/auto_configurator/auto_config.py \
    --model_type=llama \
    --log_dir=/workspace/examples/llm/auto_configurator/auto_conf_logs_llama \
    --run_number=3

export CUDA_VISIBLE_DEVICES=0,1
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo -m torch.distributed.launch --nproc_per_node=2 --use-env examples/llm/auto_configurator/auto_config.py \
    --model_type=llama \
    --log_dir=/workspace/examples/llm/auto_configurator/auto_conf_logs_llama \
    --get_results
