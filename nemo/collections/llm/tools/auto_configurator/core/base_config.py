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

"""Generates base configuration for given model."""

import math
import os
from typing import Tuple

from nemo.collections.llm.tools.auto_configurator.core.utils import generic_base_config


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
    valid_models = ["gpt3", "t5", "mt5", "bert", "llama", "mixtral", "mistral"]
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
    valid_models = ["gpt3", "t5", "mt5", "bert", "llama", "mixtral", "mistral"]
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


def generate_base_config(
    model_name: str,
    model_version: int,
    model_size_in_b: int,
    model_measure: str,
    cfg: dict,
):
    """
    Generates base config dictionary for a given model name and size.
    :param str model_name: name of the model, such as gpt3, t5, mt5...
    :param int model_version: version of model to be trained.
    :param float model_size_in_b: number of parameters in the model, if known.
    :param str model_measure: measure of model size (millions or billions).
    :param dict cfg: full config object.
    :return: base config object for the given model.
    :rtype: dict
    """

    base_cfg = generic_base_config(
        model_name=model_name,
        model_version=model_version,
        model_size_in_b=model_size_in_b,
        model_measure=model_measure,
        cfg=cfg,
    )

    return base_cfg
