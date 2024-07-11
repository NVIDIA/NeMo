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

import os
from typing import Optional

import omegaconf
from autoconfig.base_config import calculate_model_size, generate_base_config
from autoconfig.inference_sweep import search_inference_config
from autoconfig.training_config import search_training_config

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
]


def search_config(
    cfg: omegaconf.dictconfig.DictConfig, hydra_args: Optional[str] = None
):
    """
    Main function that implements the entire pipeline to search the optimal
    model config and launch the grid searches for both training and inference
    constraints.
    :param omegaconf.dictconfig.DictConfig cfg: main hydra config object for the HP tool.
    :param Optional[str] hydra_args: hydra override arguments in string format.
    :return: None
    """
    model_type = cfg.get("search_config_value")
    model_name, model_size = model_type.split("/")
    assert (
        model_name in SUPPORTED_MODELS
    ), f"search_config must be set to one of {SUPPORTED_MODELS}/<model_size>"

    # Read config
    hp_cfg = cfg.get("search_config")
    train_cfg = hp_cfg.get("train_settings")
    nodes = train_cfg.get("num_nodes")
    gpus_per_node = train_cfg.get("gpus_per_node")
    gpu_memory_gb = train_cfg.get("gpu_memory_gb")
    max_training_days = train_cfg.get("max_training_days")
    max_minutes_per_run = train_cfg.get("max_minutes_per_run")
    model_size_in_b = train_cfg.get("model_size_in_b")
    vocab_size = train_cfg.get("vocab_size")
    tflops_per_gpu = train_cfg.get("tflops_per_gpu")
    num_tokens_in_b = train_cfg.get("num_tokens_in_b")
    seq_length = train_cfg.get("seq_length")
    custom_cfg = train_cfg.get("custom_config")

    gpu_count = nodes * gpus_per_node
    assert (
        isinstance(gpu_count, int) and gpu_count > 0
    ), "nodes * gpus_per_node must be an int larger than zero."
    assert isinstance(gpu_memory_gb, int) and gpu_memory_gb in (
        40,
        80,
    ), "gpu_memory_gb can only be 40 or 80."
    assert (
        isinstance(max_minutes_per_run, int) and max_minutes_per_run >= 10
    ), "max_minutes_per_run must be an int and be at least 10 minutes."

    # Logging config
    log_dir = train_cfg.get("logs")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "candidate_configs"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "training_logs"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "final_result"), exist_ok=True)

    # Calculate model size
    model_size_in_b = calculate_model_size(
        gpu_count=gpu_count,
        max_training_days=max_training_days,
        model_size_in_b=model_size_in_b,
        tflops_per_gpu=tflops_per_gpu,
        num_tokens_in_b=num_tokens_in_b,
        model_name=model_name,
    )
    cfg.search_config.train_settings.model_size_in_b = model_size_in_b

    # Generate base config for the given model size
    base_cfg = generate_base_config(
        model_size_in_b=model_size_in_b,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        gpu_memory_gb=gpu_memory_gb,
        max_training_days=max_training_days,
        num_tokens_in_b=num_tokens_in_b,
        vocab_size=vocab_size,
        seq_length=seq_length,
        custom_cfg=custom_cfg,
        cfg=cfg,
        model_name=model_name,
    )

    # Launch grid search for training constraints
    if cfg.get("run_training_hp_search"):
        search_training_config(base_cfg, model_size_in_b, model_name, hydra_args, cfg)

    # Launch grid search for inference constraints
    if cfg.get("run_inference_hp_search"):
        search_inference_config(
            base_cfg=base_cfg, cfg=cfg,
        )
