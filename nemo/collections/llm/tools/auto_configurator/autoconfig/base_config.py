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

"""Generate base YAML configuration for any model type and size."""

import math
import os
from typing import Tuple

import omegaconf
import yaml
from autoconfig import utils


def calculate_model_size(
    gpu_count: int,
    max_training_days: float,
    model_size_in_b: float = None,
    tflops_per_gpu: int = 140,
    num_tokens_in_b: int = 300,
    model_name: str = "gpt3",
) -> float:
    """
    Estimates a model size to be trained given the constraints. If the
    model_size is provided, it estimates the time to train it with the given
    constraints.
    Example: output 5B params to train for 7 days with 160 GPUs.
    :param int gpu_count: number of gpus to use (num_nodes * gpus_per_node).
    :param float max_training_days: number of days to train the model for.
    :param float model_size_in_b: number of parameters in the model, if known.
    :param int tflops_per_gpu: estimated number of TFLOPS/s per GPU.
    :param int num_tokens_in_b: number of tokens to train the model for.
    :return: number of parameters to use for training.
    :rtype: float
    """
    # Model size is not known, must be estimated.
    if model_size_in_b is None:
        model_size_in_b = _estimate_model_size(
            max_training_days=max_training_days,
            gpu_count=gpu_count,
            tflops_per_gpu=tflops_per_gpu,
            num_tokens_in_b=num_tokens_in_b,
            model_name=model_name,
        )
    # Model size is known, so only time to train estimate is needed.
    else:
        max_training_days = _estimate_training_time(
            model_size_in_b=model_size_in_b,
            gpu_count=gpu_count,
            tflops_per_gpu=tflops_per_gpu,
            num_tokens_in_b=num_tokens_in_b,
            model_name=model_name,
        )

    print(
        f"You can train a {model_size_in_b}B parameter model in "
        f"{max_training_days} days using {gpu_count} GPUs. This result assumes "
        f"you are training to {num_tokens_in_b}B tokens, and each GPU achieves "
        f"{tflops_per_gpu} TFLOPS."
    )
    return model_size_in_b


def _estimate_model_size(
    max_training_days: float,
    gpu_count: int,
    tflops_per_gpu: int,
    num_tokens_in_b: int,
    model_name: str,
) -> float:
    """
    Estimates model size given time and hardware constraints. It's only used if the model size is
    not provided by the user.
    :param float max_training_days: number of days to train the model for.
    :param int gpu_count: number of gpus to use (num_nodes * gpus_per_node).
    :param int tflops_per_gpu: estimated number of TFLOPS/s per GPU.
    :param int num_tokens_in_b: number of tokens to train the model for.
    :param str model_name: name of the model, such as gpt3, t5, mt5...
    :return: number of parameters to use for training.
    :rtype: float
    :raises NotImplementedError: if the model_name is not one of the supported models.
    """
    model_penalty = 0.87 if model_name == "mt5" else 1.0
    valid_models = ["gpt3", "t5", "mt5", "bert"]
    try:
        if model_name in valid_models:
            return round(
                model_penalty
                * (max_training_days * 3600 * 24 * gpu_count * tflops_per_gpu * 1e12)
                / (8 * num_tokens_in_b * 1e9)
                / 1e9,
                2,
            )
        else:
            raise NotImplementedError
    except ValueError as err:
        print(f"Input values were not valid: {err}")
    except ZeroDivisionError as err:
        print(f"Cannot divide by zero. This can happen if num_tokens_in_b is zero: {err}")
    except NotImplementedError as err:
        print(f"Model size estimation is only available for {valid_models}: {err}")
    return None


def _estimate_training_time(
    model_size_in_b: float,
    gpu_count: int,
    tflops_per_gpu: int,
    num_tokens_in_b: int,
    model_name: str,
) -> float:
    """
    Estimates training time for a given model size and hardware constraint. To be used when
    a model size is provided by the user.
    :param float model_size_in_b: number of parameters to use for training.
    :param int gpu_count: number of gpus to use (num_nodes * gpus_per_node).
    :param int tflops_per_gpu: estimated number of TFLOPS/s per GPU.
    :param int num_tokens_in_b: number of tokens to train the model for.
    :param str model_name: name of the model, such as gpt3, t5, mt5...
    :return: number of days it will take to train the model.
    :rtype: float
    :raises NotImplementedError: if the model_name is not one of the supported models.
    """
    model_penalty = 1.15 if model_name == "mt5" else 1.0
    valid_models = ["gpt3", "t5", "mt5", "bert"]
    try:
        if model_name in valid_models:
            return round(
                model_penalty
                * (model_size_in_b * 1e9 * 8 * num_tokens_in_b * 1e9)
                / (3600 * 24 * gpu_count * tflops_per_gpu * 1e12),
                2,
            )
        else:
            raise NotImplementedError
    except ValueError as err:
        print(f"Input values were not valid: {err}")
    except ZeroDivisionError as err:
        print(f"Cannot divide by zero. This can happen if gpu_count or tflops_per_gpu are zero: {err}")
    except NotImplementedError as err:
        print(f"Training time estimation is only available for {valid_models}: {err}")
    return None


def _calculate_gbs_tp_pp(
    model_size_in_b: float,
    seq_length: int,
    gpu_memory_gb: int = 80,
    model_name: str = "gpt3",
) -> Tuple[int]:
    """
    Calculates Global Batch Size (GBS), Tensor Parallelism (TP), and Pipeline
    Parallelism (PP) values, given a model size and model name.
    :param float model_size_in_b: the number of parameters in the model.
    :param int gpu_memory_gb: memory available per GPU, in GBs.
    :param str model_name: name of the model, such as gpt3, t5, mt5...
    :returns: tuple (gbs, tp, pp)
        WHERE
        int gbs is the Global Batch Size to use for training.
        int tp is the Tensor Parallelism value to use for training.
        int pp is the Pipeline Parallelism value to use for training.
    :raises NotImplementedError: if the model_name is not one of the supported models.
    """
    if model_name == "gpt3":
        if gpu_memory_gb == 80:
            return _gbs_tp_pp_gpt3_80gb(model_size_in_b=model_size_in_b, seq_length=seq_length)
        elif gpu_memory_gb == 40:
            return _gbs_tp_pp_gpt3_40gb(model_size_in_b=model_size_in_b, seq_length=seq_length)
    elif model_name in ["t5", "mt5"]:
        if gpu_memory_gb == 80:
            return _gbs_tp_pp_t5_80gb(model_size_in_b=model_size_in_b, seq_length=seq_length)
        elif gpu_memory_gb == 40:
            return _gbs_tp_pp_t5_40gb(model_size_in_b=model_size_in_b, seq_length=seq_length)
    elif model_name == "bert":
        if gpu_memory_gb == 80:
            return _gbs_tp_pp_bert_80gb(model_size_in_b=model_size_in_b, seq_length=seq_length)
        elif gpu_memory_gb == 40:
            return _gbs_tp_pp_bert_40gb(model_size_in_b=model_size_in_b, seq_length=seq_length)
    else:
        raise NotImplementedError("Only gpt3, t5, mt5 and bert are supported.")
    return None


def _gbs_tp_pp_gpt3_80gb(model_size_in_b: float, seq_length: int) -> Tuple[int]:
    """
    Outputs GBS, TP and PP values for any GPT-3 model size for 80GB GPUs.
    :param float model_size_in_b: the number of parameters in the model.
    :returns: tuple (gbs, tp, pp, cp, ep)
        WHERE
        int gbs is the Global Batch Size to use for training.
        int tp is the Tensor Parallelism value to use for training.
        int pp is the Pipeline Parallelism value to use for training.
        int cp is the Context Parallelism value to use for training.
        int ep is the Expert Parallelism value to use for training.
    :raises ValueError: if the model_size_in_b is larger than the supported max model size.
    """
    cp = 1
    ep = 1
    if seq_length == 2048:
        if model_size_in_b <= 1.0:
            gbs, tp, pp = 256, 1, 1
        elif model_size_in_b <= 4.0:
            gbs, tp, pp = 1024, 1, 1
        elif model_size_in_b <= 8.0:
            gbs, tp, pp = 2048, 2, 1
        elif model_size_in_b <= 13.0:
            gbs, tp, pp = 2048, 4, 1
        elif model_size_in_b <= 20.6:
            gbs, tp, pp = 2048, 8, 1
        elif model_size_in_b <= 45.6:
            gbs, tp, pp = 2048, 8, 2
        elif model_size_in_b <= 123.6:
            gbs, tp, pp = 2048, 8, 4
        elif model_size_in_b <= 196.6:
            gbs, tp, pp = 2048, 8, 8
        elif model_size_in_b <= 392.2:
            gbs, tp, pp = 2048, 8, 16
        elif model_size_in_b <= 735:
            gbs, tp, pp = 2048, 8, 32
        elif model_size_in_b <= 1100:
            gbs, tp, pp = 2048, 8, 64
        else:
            raise ValueError("No GPT-3 model larger than 1.1T parameters is supported.")
    elif seq_length == 4096:
        if model_size_in_b <= 1.0:
            gbs, tp, pp = 128, 1, 1
        elif model_size_in_b <= 4.0:
            gbs, tp, pp = 512, 1, 1
        elif model_size_in_b <= 8.0:
            gbs, tp, pp = 1024, 2, 1
        elif model_size_in_b <= 13.0:
            gbs, tp, pp = 1024, 4, 1
        elif model_size_in_b <= 20.6:
            gbs, tp, pp = 1024, 4, 2
        elif model_size_in_b <= 45.6:
            gbs, tp, pp = 1024, 8, 2
        else:
            raise ValueError("No GPT-3 model larger than 45.6B parameters is supported with sequnce length 4096.")
    elif seq_length == 8192:
        if model_size_in_b <= 1.0:
            gbs, tp, pp = 64, 1, 1
        elif model_size_in_b <= 4.0:
            gbs, tp, pp = 256, 1, 1
        elif model_size_in_b <= 8.0:
            gbs, tp, pp = 512, 2, 1
        elif model_size_in_b <= 13.0:
            gbs, tp, pp = 512, 4, 1
        elif model_size_in_b <= 20.6:
            gbs, tp, pp = 512, 4, 4
        elif model_size_in_b <= 45.6:
            gbs, tp, pp = 512, 8, 2
        else:
            raise ValueError("No GPT-3 model larger than 45.6B parameters is supported with sequnce length 8192.")
    elif seq_length == 16384:
        if model_size_in_b <= 1.0:
            gbs, tp, pp = 32, 2, 1
        elif model_size_in_b <= 4.0:
            gbs, tp, pp = 128, 2, 1
        elif model_size_in_b <= 8.0:
            gbs, tp, pp = 256, 2, 2
        elif model_size_in_b <= 13.0:
            gbs, tp, pp = 256, 4, 1
        elif model_size_in_b <= 20.6:
            gbs, tp, pp = 256, 8, 2
        else:
            raise ValueError("No GPT-3 model larger than 20.6B parameters is supported with sequnce length 16384.")
    elif seq_length == 32768:
        if model_size_in_b <= 1.0:
            gbs, tp, pp = 16, 2, 1
        elif model_size_in_b <= 4.0:
            gbs, tp, pp = 64, 2, 1
        elif model_size_in_b <= 8.0:
            gbs, tp, pp = 128, 4, 2
        elif model_size_in_b <= 13.0:
            gbs, tp, pp = 128, 4, 2
        elif model_size_in_b <= 20.6:
            gbs, tp, pp = 128, 8, 2
        else:
            raise ValueError("No GPT-3 model larger than 20.6B parameters is supported with sequnce length 32768.")
    else:
        raise ValueError(
            f"seq_length = {seq_length} is not supported. Available seq_length list for GPT-3 models: [2048, 4096, 8192, 16384, 32768]"
        )
    return gbs, tp, pp, cp, ep


def _gbs_tp_pp_gpt3_40gb(model_size_in_b: float, seq_length: int) -> Tuple[int, int, int]:
    """
    Outputs GBS, TP and PP values for any GPT-3 model size for 40GB GPUs.
    :param float model_size_in_b: the number of parameters in the model.
    :returns: tuple (gbs, tp, pp, cp, ep)
        WHERE
        int gbs is the Global Batch Size to use for training.
        int tp is the Tensor Parallelism value to use for training.
        int pp is the Pipeline Parallelism value to use for training.
        int cp is the Context Parallelism value to use for training.
        int ep is the Expert Parallelism value to use for training.
    :raises ValueError: if the model_size_in_b is larger than the supported max model size.
    """
    cp = 1
    ep = 1
    if seq_length == 2048:
        if model_size_in_b <= 1.0:
            gbs, tp, pp = 256, 1, 1
        elif model_size_in_b <= 4.0:
            gbs, tp, pp = 1024, 4, 1
        elif model_size_in_b <= 8.0:
            gbs, tp, pp = 2048, 8, 1
        elif model_size_in_b <= 13.0:
            gbs, tp, pp = 2048, 8, 2
        elif model_size_in_b <= 20.6:
            gbs, tp, pp = 2048, 8, 4
        elif model_size_in_b <= 45.6:
            gbs, tp, pp = 2048, 8, 4
        elif model_size_in_b <= 123.6:
            gbs, tp, pp = 2048, 8, 8
        elif model_size_in_b <= 196.6:
            gbs, tp, pp = 2048, 8, 16
        elif model_size_in_b <= 392.2:
            gbs, tp, pp = 2048, 8, 32
        elif model_size_in_b <= 735:
            gbs, tp, pp = 2048, 8, 64
        elif model_size_in_b <= 1100:
            gbs, tp, pp = 2048, 8, 128
        else:
            raise ValueError("No GPT-3 model larger than 1.1T parameters is supported.")
    else:
        raise ValueError("seq_length != 2048 is not supported on 40GB GPU.")
    return gbs, tp, pp, cp, ep


def _gbs_tp_pp_t5_80gb(model_size_in_b: float, seq_length: int) -> Tuple[int, int, int]:
    """
    Outputs GBS, TP and PP values for any T5/mT5 model size for 80GB GPUs.
    :param float model_size_in_b: the number of parameters in the model.
    :returns: tuple (gbs, tp, pp, cp, ep)
        WHERE
        int gbs is the Global Batch Size to use for training.
        int tp is the Tensor Parallelism value to use for training.
        int pp is the Pipeline Parallelism value to use for training.
        int cp is the Context Parallelism value to use for training.
        int ep is the Expert Parallelism value to use for training.
    :raises ValueError: if the model_size_in_b is larger than the supported max model size.
    """
    cp = None
    ep = None
    if seq_length == 512:
        if model_size_in_b <= 1.0:
            gbs, tp, pp = 2048, 1, 1
        elif model_size_in_b <= 5.0:
            gbs, tp, pp = 1920, 2, 1
        elif model_size_in_b <= 11.5:
            gbs, tp, pp = 1920, 4, 1
        elif model_size_in_b <= 18.5:
            gbs, tp, pp = 1920, 8, 1
        elif model_size_in_b <= 25.9:
            gbs, tp, pp = 1920, 8, 2
        elif model_size_in_b <= 43.0:
            gbs, tp, pp = 1920, 8, 4
        elif model_size_in_b <= 85.5:
            gbs, tp, pp = 1920, 8, 8
        elif model_size_in_b <= 165.5:
            gbs, tp, pp = 1920, 8, 16
        elif model_size_in_b <= 250:
            gbs, tp, pp = 1920, 8, 32
        else:
            raise ValueError("No T5/mT5 model larger than 250B parameters is supported.")
    else:
        raise ValueError(f"seq_length = {seq_length} is not supported. Available seq_length list for T5 models: [512]")
    return gbs, tp, pp, cp, ep


def _gbs_tp_pp_t5_40gb(model_size_in_b: float, seq_length: int) -> Tuple[int, int, int]:
    """
    Outputs GBS, TP and PP values for any T5/mT5 model size for 40GB GPUs.
    :param float model_size_in_b: the number of parameters in the model.
    :returns: tuple (gbs, tp, pp, cp, ep)
        WHERE
        int gbs is the Global Batch Size to use for training.
        int tp is the Tensor Parallelism value to use for training.
        int pp is the Pipeline Parallelism value to use for training.
        int cp is the Context Parallelism value to use for training.
        int ep is the Expert Parallelism value to use for training.
    :raises ValueError: if the model_size_in_b is larger than the supported max model size.
    """
    cp = None
    ep = None
    if seq_length == 512:
        if model_size_in_b <= 0.5:
            gbs, tp, pp = 2048, 1, 1
        if model_size_in_b <= 1.0:
            gbs, tp, pp = 2048, 2, 1
        elif model_size_in_b <= 5.0:
            gbs, tp, pp = 1920, 4, 1
        elif model_size_in_b <= 11.5:
            gbs, tp, pp = 1920, 8, 1
        elif model_size_in_b <= 18.5:
            gbs, tp, pp = 1920, 8, 2
        elif model_size_in_b <= 25.9:
            gbs, tp, pp = 1920, 8, 4
        elif model_size_in_b <= 43.0:
            gbs, tp, pp = 1920, 8, 8
        elif model_size_in_b <= 85.5:
            gbs, tp, pp = 1920, 8, 16
        elif model_size_in_b <= 165.5:
            gbs, tp, pp = 1920, 8, 32
        elif model_size_in_b <= 250:
            gbs, tp, pp = 1920, 8, 64
        else:
            raise ValueError("No T5/mT5 model larger than 250B parameters is supported.")
    else:
        raise ValueError(f"seq_length = {seq_length} is not supported. Available seq_length list for T5 models: [512]")
    return gbs, tp, pp, cp, ep


def _gbs_tp_pp_bert_80gb(model_size_in_b: float, seq_length: int) -> Tuple[int, int, int]:
    """
    Outputs GBS, TP and PP values for any BERT model size for 80GB GPUs.
    :param float model_size_in_b: the number of parameters in the model.
    :returns: tuple (gbs, tp, pp, cp, ep)
        WHERE
        int gbs is the Global Batch Size to use for training.
        int tp is the Tensor Parallelism value to use for training.
        int pp is the Pipeline Parallelism value to use for training.
        int cp is the Context Parallelism value to use for training.
        int ep is the Expert Parallelism value to use for training.
    :raises ValueError: if the model_size_in_b is larger than the supported max model size.
    """
    cp = None
    ep = None
    if seq_length == 512:
        if model_size_in_b <= 1.0:
            gbs, tp, pp = 256, 1, 1
        elif model_size_in_b <= 3.2:
            gbs, tp, pp = 1024, 1, 1
        elif model_size_in_b <= 8.0:
            gbs, tp, pp = 2048, 2, 1
        elif model_size_in_b <= 13.0:
            gbs, tp, pp = 2048, 4, 1
        elif model_size_in_b <= 25.5:
            gbs, tp, pp = 2048, 8, 1
        elif model_size_in_b <= 46.5:
            gbs, tp, pp = 2048, 8, 2
        elif model_size_in_b <= 87.5:
            gbs, tp, pp = 2048, 8, 4
        elif model_size_in_b <= 165.5:
            gbs, tp, pp = 4096, 8, 8
        elif model_size_in_b <= 250.5:
            gbs, tp, pp = 2048, 8, 16
        else:
            raise ValueError("No BERT model larger than 250B parameters is supported.")
    else:
        raise ValueError(
            f"seq_length = {seq_length} is not supported. Available seq_length list for BERT models: [512]"
        )
    return gbs, tp, pp, cp, ep


def _gbs_tp_pp_bert_40gb(model_size_in_b: float, seq_length: int) -> Tuple[int, int, int]:
    """
    Outputs GBS, TP and PP values for any BERT model size for 40GB GPUs.
    :param float model_size_in_b: the number of parameters in the model.
    :returns: tuple (gbs, tp, pp, cp, ep)
        WHERE
        int gbs is the Global Batch Size to use for training.
        int tp is the Tensor Parallelism value to use for training.
        int pp is the Pipeline Parallelism value to use for training.
        int cp is the Context Parallelism value to use for training.
        int ep is the Expert Parallelism value to use for training.
    :raises ValueError: if the model_size_in_b is larger than the supported max model size.
    """
    cp = None
    ep = None
    if seq_length == 512:
        if model_size_in_b <= 1.0:
            gbs, tp, pp = 256, 1, 1
        elif model_size_in_b <= 3.2:
            gbs, tp, pp = 1024, 4, 1
        elif model_size_in_b <= 8.0:
            gbs, tp, pp = 2048, 8, 1
        elif model_size_in_b <= 13.0:
            gbs, tp, pp = 2048, 8, 2
        elif model_size_in_b <= 25:
            gbs, tp, pp = 2048, 8, 4
        elif model_size_in_b <= 46.5:
            gbs, tp, pp = 2048, 8, 8
        elif model_size_in_b <= 87.5:
            gbs, tp, pp = 2048, 8, 16
        elif model_size_in_b <= 165.5:
            gbs, tp, pp = 2048, 8, 32
        elif model_size_in_b <= 250.5:
            gbs, tp, pp = 2048, 8, 64
        else:
            raise ValueError("No BERT model larger than 250B parameters is supported.")
    else:
        raise ValueError(
            f"seq_length = {seq_length} is not supported. Available seq_length list for BERT models: [512]"
        )
    return gbs, tp, pp, cp, ep


def generate_base_config(
    model_size_in_b: float,
    nodes: int,
    gpus_per_node: int,
    gpu_memory_gb: int,
    max_training_days: float,
    num_tokens_in_b: int,
    vocab_size: int,
    seq_length: int,
    custom_cfg: str,
    model_name: str,
    cfg: omegaconf.dictconfig.DictConfig,
):
    """
    Generates base config dictionary for a given model name and size.
    :param float model_size_in_b: number of parameters in the model, if known.
    :param int nodes: number of nodes to use for training.
    :param int gpus_per_node: number of GPUs available in each node.
    :param float max_training_days: number of days to train the model for.
    :param int num_tokens_in_b: number of tokens to train the model for.
    :param str model_name: name of the model, such as gpt3, t5, mt5...
    :param omegaconf.dictconfig.DictConfig cfg: full config object.
    :return: base config object for the given model.
    :rtype: dict
    """
    base_cfg = utils.generic_base_config(cfg=cfg, custom_cfg=custom_cfg, model_name=model_name)

    # GBS: global batch size
    if custom_cfg is None:
        gbs, tp, pp, cp, ep = _calculate_gbs_tp_pp(
            model_size_in_b=model_size_in_b,
            gpu_memory_gb=gpu_memory_gb,
            model_name=model_name,
            seq_length=seq_length,
        )
    else:
        gbs = base_cfg["model"]["global_batch_size"]
        tp = base_cfg["model"]["tensor_model_parallel_size"]
        pp = base_cfg["model"]["pipeline_model_parallel_size"]
        default_cp = None if model_name in ["bert", "t5", "mt5"] else 1
        default_ep = None if model_name in ["bert", "t5", "mt5"] else 1
        cp = base_cfg["model"].get("context_parallel_size", default_cp)
        ep = base_cfg["model"].get("expert_model_parallel_size", default_ep)
    # RUN
    base_cfg["run"]["name"] = f"{model_name}_{model_size_in_b}b"
    base_cfg["run"]["results_dir"] = "${base_results_dir}/${.name}"
    int_days = int(max_training_days)
    int_hours = int(24 * (max_training_days - int(max_training_days)))
    base_cfg["run"]["time_limit"] = f"{int_days}-{int_hours:02d}:00:00"

    # TRAINER
    base_cfg["trainer"]["num_nodes"] = nodes
    base_cfg["trainer"]["precision"] = "bf16"
    base_cfg["trainer"]["max_steps"] = int((num_tokens_in_b * 1e9) / (seq_length * gbs))
    if int_hours == 0:
        int_days -= 1
        int_hours = 23
    else:
        int_hours -= 1
    base_cfg["trainer"]["max_time"] = f"{int_days}:{int_hours:02d}:30:00"

    # EXP_MANAGER
    wandb_cfg = cfg.get("wandb")
    enable = wandb_cfg.get("enable")
    project = wandb_cfg.get("project")
    if enable:
        base_cfg["exp_manager"]["create_wandb_logger"] = bool(enable)
        base_cfg["exp_manager"]["wandb_logger_kwargs"]["project"] = project

    # MODEL
    if custom_cfg is None:
        layers, hs, att_h, ffn, kv, lr = utils.calculate_model_size_params(
            model_size_in_b=model_size_in_b,
            vocab_size=vocab_size,
            seq_length=seq_length,
            model_name=model_name,
        )
        if model_name == "gpt3":
            base_cfg["model"]["num_layers"] = int(layers)
            base_cfg["model"]["global_batch_size"] = int(gbs)
            base_cfg["model"]["hidden_size"] = int(hs)
            base_cfg["model"]["num_attention_heads"] = int(att_h)
            base_cfg["model"]["encoder_seq_length"] = seq_length
            base_cfg["model"]["max_position_embeddings"] = seq_length
            base_cfg["model"]["data"]["seq_length"] = seq_length
            if ffn is not None:
                base_cfg["model"]["ffn_hidden_size"] = int(ffn)
            if kv is not None:
                base_cfg["model"]["kv_channels"] = int(kv)
            base_cfg["model"]["init_method_std"] = round(0.64 / math.sqrt(hs), 6)
            base_cfg["model"]["optim"]["sched"]["warmup_steps"] = int(0.0015 * base_cfg["trainer"]["max_steps"])
            base_cfg["model"]["optim"]["sched"]["constant_steps"] = int(0.166 * base_cfg["trainer"]["max_steps"])
            if model_size_in_b <= 13.0:
                base_cfg["model"]["sequence_parallel"] = False
        elif model_name == "bert":
            base_cfg["model"]["global_batch_size"] = int(gbs)
            base_cfg["model"]["num_layers"] = int(layers)
            base_cfg["model"]["hidden_size"] = int(hs)
            base_cfg["model"]["num_attention_heads"] = int(att_h)
            if ffn is not None:
                base_cfg["model"]["ffn_hidden_size"] = int(ffn)
            if kv is not None:
                base_cfg["model"]["kv_channels"] = int(kv)
            base_cfg["model"]["init_method_std"] = round(0.64 / math.sqrt(hs), 6)
            base_cfg["model"]["optim"]["sched"]["warmup_steps"] = int(0.0015 * base_cfg["trainer"]["max_steps"])
            base_cfg["model"]["optim"]["sched"]["constant_steps"] = int(0.166 * base_cfg["trainer"]["max_steps"])
            if model_size_in_b <= 13.0:
                base_cfg["model"]["sequence_parallel"] = False
        else:
            base_cfg["model"]["global_batch_size"] = int(gbs)
            base_cfg["model"]["encoder"]["num_layers"] = int(layers)
            base_cfg["model"]["decoder"]["num_layers"] = int(layers)
            base_cfg["model"]["encoder"]["hidden_size"] = int(hs)
            base_cfg["model"]["decoder"]["hidden_size"] = int(hs)
            base_cfg["model"]["encoder"]["num_attention_heads"] = int(att_h)
            base_cfg["model"]["decoder"]["num_attention_heads"] = int(att_h)
            if ffn is not None:
                base_cfg["model"]["encoder"]["ffn_hidden_size"] = int(ffn)
                base_cfg["model"]["decoder"]["ffn_hidden_size"] = int(ffn)
            if kv is not None:
                base_cfg["model"]["encoder"]["kv_channels"] = int(kv)
                base_cfg["model"]["decoder"]["kv_channels"] = int(kv)
            base_cfg["model"]["init_method_std"] = 0.015
            base_cfg["model"]["optim"]["sched"]["warmup_ratio"] = 0.01

        base_cfg["model"]["optim"]["lr"] = lr
        base_cfg["model"]["optim"]["sched"]["min_lr"] = round(lr * 0.1, 8)

    if cfg.get("cluster_type") == "bcp":
        index_map_dir = os.path.join(cfg.get("base_results_dir"), "data_index_files")
        os.makedirs(index_map_dir, exist_ok=True)
        base_cfg["model"]["data"]["index_mapping_dir"] = index_map_dir

    with open(f"{cfg.search_config.train_settings.logs}/base_cfg_{model_size_in_b}b.yaml", "w") as f:
        yaml.dump(base_cfg, f)
    return base_cfg
