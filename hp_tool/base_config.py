import os
import time
import math

import subprocess
import submitit
import yaml
import omegaconf

from hp_tool import utils


def calculate_model_size(
    gpu_count,
    max_training_days,
    model_size_in_b=None,
    tflops_per_gpu=140,
    num_tokens_in_b=300,
):
    """
    Estimates a model size to be trained given the constraints. If the
    model_size is provided, it estimates the time to train it with the given
    constraints.

    Example: output 5B params to train for 7 days with 160 GPUs.

    Arguments:
        gpu_count: int, number of gpus to use (num_nodes * gpus_per_node).
        max_training_days: float, number of days to train the model for.
        model_size_in_b: float, number of parameters in the model, if known.
        tflops_per_gpu: int, estimated number of TFLOPS/s per GPU.
        num_tokens_in_b: int, number of tokens to train the model for.
    Output:
        model_size_in_b: int, number of parameters to use for training.
    """
    assert (
        isinstance(gpu_count, int) and gpu_count > 0
    ), "gpu_count must be an int larger than zero."
    assert isinstance(max_training_days, float) or isinstance(
        max_training_days, int
    ), "max_training_days must be int or float."
    assert max_training_days > 0, "max_training_days must be larger than zero."
    assert (
        isinstance(tflops_per_gpu, int) and tflops_per_gpu > 0
    ), "tflops_per_gpu must be an int larger than zero."
    assert (
        isinstance(num_tokens_in_b, int) and num_tokens_in_b > 0
    ), "num_tokens_in_b must be an int larger than zero."

    if model_size_in_b is None:
        model_size_in_b = round(
            (max_training_days * 3600 * 24 * gpu_count * tflops_per_gpu * 1e12)
            / (8 * num_tokens_in_b * 1e9) / 1e9,
            2,
        )
    else:
        assert isinstance(model_size_in_b, float) or isinstance(
            model_size_in_b, int
        ), "model_size_in_b must be int or float."
        assert model_size_in_b > 0, "model_size_in_b must be larger than zero."
        max_training_days = round(
            (model_size_in_b * 1e9 * 8 * num_tokens_in_b * 1e9)
            / (3600 * 24 * gpu_count * tflops_per_gpu * 1e12),
            2,
        )
    print(
        f"You can train a {model_size_in_b}B parameter model in "
        f"{max_training_days} days using {gpu_count} GPUs. This result assumes "
        f"you are training to {num_tokens_in_b}B tokens, and each GPU achieves "
        f"{tflops_per_gpu} TFLOPS."
    )
    return model_size_in_b


def _calculate_gbs_tp_pp(model_size_in_b):
    """
    Calculates Global Batch Size (GBS), Tensor Parallelism (TP), and Pipeline 
    Parallelism (PP) values, given a model size.

    Arguments:
        model_size_in_b: float, the number of parameters in the model.
    Output:
        gbs: int, global batch size to use for training.
        tp: int, tensor parallelism to use for training.
        pp: int, pipeline parallelism to use for training.
    """
    if model_size_in_b <= 1.0:
        gbs, tp, pp = 256, 1, 1
    elif model_size_in_b <= 4.0:
        gbs, tp, pp = 720, 1, 1
    elif model_size_in_b <= 8.0:
        gbs, tp, pp = 1440, 2, 1
    elif model_size_in_b <= 13.0:
        gbs, tp, pp = 1440, 4, 1
    elif model_size_in_b <= 20.6:
        gbs, tp, pp = 1440, 8, 1
    elif model_size_in_b <= 45.6:
        gbs, tp, pp = 1440, 8, 4
    elif model_size_in_b <= 123.6:
        gbs, tp, pp = 1440, 8, 8
    elif model_size_in_b <= 196.6:
        gbs, tp, pp = 1536, 8, 16
    elif model_size_in_b <= 392.2:
        gbs, tp, pp = 1792, 8, 32
    elif model_size_in_b <= 735:
        gbs, tp, pp = 1920, 8, 64
    elif model-size_in_b <= 1100:
        gbs, tp, pp = 2048, 8, 128
    else:
        print("No model larger than 1.1T parameters is supported.")
        raise ValueError
    return gbs, tp, pp


def generate_base_config(
    model_size_in_b, nodes, gpus_per_node, max_training_days, num_tokens_in_b, cfg
):
    # GBS: global batch size
    gbs, tp, pp = _calculate_gbs_tp_pp(model_size_in_b=model_size_in_b)

    base_cfg = utils.generic_base_config(cfg.search_config)

    # RUN
    base_cfg["run"]["name"] = f"{model_size_in_b}b"
    base_cfg["run"]["results_dir"] = "${base_results_dir}/${.name}"
    base_cfg["run"]["time_limit"] = (
        f"{int(max_training_days)}-"
        f"{int(24 * (max_training_days - int(max_training_days)))}:00:00"
    )

    # TRAINER
    base_cfg["trainer"]["precision"] = "bf16" if model_size_in_b <= 5.5 else "bf16"
    mbs = base_cfg["model"]["micro_batch_size"]
    seq_length = base_cfg["model"]["data"]["seq_length"]
    base_cfg["trainer"]["max_steps"] = int((num_tokens_in_b * 1e9) / (seq_length * gbs))
    base_cfg["trainer"]["max_time"] = (
        f"{int(max_training_days)}:"
        f"{int(24 * (max_training_days - int(max_training_days))) - 1}:40:00"
    )

    # MODEL
    layers, hs, att_heads, lr = utils.calculate_layers_hs_lr(model_size_in_b=model_size_in_b)
    base_cfg["model"]["num_layers"] = int(layers)
    base_cfg["model"]["hidden_size"] = int(hs)
    base_cfg["model"]["num_attention_heads"] = int(att_heads)
    base_cfg["model"]["init_method_std"] = round(0.64 / math.sqrt(hs), 6)
    base_cfg["model"]["optim"]["lr"] = lr
    base_cfg["model"]["optim"]["sched"]["min_lr"] = round(lr * 0.1, 8)
    base_cfg["model"]["optim"]["sched"]["warmup_steps"] = int(
        0.0015 * base_cfg["trainer"]["max_steps"]
    )
    base_cfg["model"]["optim"]["sched"]["constant_steps"] = int(
        0.166 * base_cfg["trainer"]["max_steps"]
    )

    with open(f"{cfg.base_results_dir}/base_cfg_{model_size_in_b}b.yaml", "w") as f:
        yaml.dump(base_cfg, f)
    return base_cfg
