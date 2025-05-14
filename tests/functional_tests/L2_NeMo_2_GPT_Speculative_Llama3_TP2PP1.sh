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
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/test_hf_import.py --hf_model /home/TestData/nlp/megatron_llama/llama-ci-hf --output_path /tmp/nemo2_ckpt

coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/modelopt/test_speculative_training.py \
    --name 'test_speculative_training' \
    --model_path /tmp/nemo2_ckpt \
    --tp_size 2 \
    --pp_size 1 \
    --devices 2 \
    --num_nodes 1 \
    --log_dir /tmp/speculative_logs \
    --max_steps 5 \
    --gbs 4 \
    --mbs 1 \
    --seq_length 8192 \
    --val_check_interval 5 \
    --log_interval 5 \
    --limit_val_batches 2 \
    --legacy_ckpt
