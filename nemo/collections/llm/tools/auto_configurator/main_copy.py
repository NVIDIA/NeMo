# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""Entry point, main file to run to launch jobs with the HP tool."""

import argparse

from autoconfig.search_config import search_config


def get_args():
    parser = argparse.ArgumentParser(description='Train a small GPT model using NeMo 2.0')
    parser.add_argument('--num_nodes', required=True, type=int, help="Number of nodes to use for training")
    parser.add_argument('--gpus_per_node', required=False, default=8, type=int, help="Number of GPUs per node")
    parser.add_argument('--gpu_memory_gb', required=False, default=80, type=int, help="GPU memory size")
    parser.add_argument('--max_training_days', required=False, default=2, type=int, help="Path to data file")
    parser.add_argument(
        '--max_minutes_per_run', required=False, default=30, type=int, help="Max minutes per job on cluster"
    )
    parser.add_argument('--model_type', required=True, type=str, help="Model size in billions")
    parser.add_argument('--model_size_in_b', required=True, type=int, help="Model size in billions")
    parser.add_argument('--vocab_size', required=False, default=32000, type=int, help="Size of tokenizer vocab")
    parser.add_argument('--tflops_per_gpu', required=False, default=140, type=int, help="Estimated tflops per GPU")
    parser.add_argument(
        '--num_tokens_in_b', required=False, default=300, type=int, help="Number of tokens in dataset in billions"
    )
    parser.add_argument('--seq_length', required=False, default=2048, type=int, help="Model sequence length")
    parser.add_argument('--log_dir', required=True, type=str, help="Path to results directory")

    return parser.parse_args()


def main():
    """
    Main function in the entire pipeline, it reads the config using hydra and calls search_config.
    :param omegaconf.dictconfig.DictConfig cfg: OmegaConf object, read using the @hydra.main decorator.
    :return: None
    """
    args = get_args()
    search_config(cfg=vars(args))


if __name__ == "__main__":
    main()
