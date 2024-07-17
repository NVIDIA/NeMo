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

import os
from typing import Optional

from autoconfig.training_config import search_training_config
from autoconfig.utils import generic_base_config

SUPPORTED_MODELS = [
    "gpt3",
    "t5",
    "mt5",
    "bert",
    "llama",
    "baichuan2",
    "chatglm",
    "qwen2",
    "mixtral",
    "mistral",
]


def search_config(cfg: dict):
    """
    Main function that implements the entire pipeline to search the optimal
    model config and launch the grid searches for both training and inference
    constraints.
    :param dict cfg: main hydra config object for the HP tool.
    :return: dictionary of generated configs.
    :rtype: dict
    """

    # Read config
    num_nodes = cfg.get("num_nodes")
    gpus_per_node = cfg.get("gpus_per_node", 8)
    gpu_memory_gb = cfg.get("gpu_memory_gb", 80)
    max_training_days = cfg.get("max_training_days", 2)
    max_minutes_per_run = cfg.get("max_minutes_per_run", 30)
    model_name = cfg.get("model_type")
    model_version = cfg.get("model_version")
    model_size_in_b = cfg.get("model_size")
    model_measure = cfg.get("model_measure", "B")
    vocab_size = cfg.get("vocab_size", 32000)
    tflops_per_gpu = cfg.get("tflops_per_gpu", 140)
    num_tokens_in_b = cfg.get("num_tokens_in_b", 300)
    seq_length = cfg.get("seq_length", 2048)
    global_batch_size = cfg.get("global_batch_size")

    assert model_name in SUPPORTED_MODELS, f"model must be set to one of {SUPPORTED_MODELS}/<model_size>"

    gpu_count = num_nodes * gpus_per_node
    assert isinstance(gpu_count, int) and gpu_count > 0, "num_nodes * gpus_per_node must be an int larger than zero."
    assert isinstance(gpu_memory_gb, int) and gpu_memory_gb in (
        40,
        80,
    ), "gpu_memory_gb can only be 40 or 80."
    assert (
        isinstance(max_minutes_per_run, int) and max_minutes_per_run >= 10
    ), "max_minutes_per_run must be an int and be at least 10 minutes."

    cfg["model_size_in_b"] = model_size_in_b
    cfg["gpu_count"] = gpu_count

    # Generate base config for the given model size
    base_cfg, train_cfg = generic_base_config(
        model_name=model_name,
        model_version=model_version,
        model_size_in_b=model_size_in_b,
        model_measure=model_measure,
        cfg=cfg,
    )

    # Launch grid search for training constraints
    configs = search_training_config(base_cfg, train_cfg)

    return configs
