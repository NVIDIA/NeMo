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
    model_name="gpt3",
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
    assert max_training_days is None or isinstance(max_training_days, float) or isinstance(
        max_training_days, int
    ), "max_training_days must be int or float."
    assert max_training_days is None or max_training_days > 0, "max_training_days must be larger than zero."
    assert (
        isinstance(tflops_per_gpu, int) and tflops_per_gpu > 0
    ), "tflops_per_gpu must be an int larger than zero."
    assert (
        isinstance(num_tokens_in_b, int) and num_tokens_in_b > 0
    ), "num_tokens_in_b must be an int larger than zero."

    if model_size_in_b is None:
        model_size_in_b = _estimate_model_size(
            max_training_days, gpu_count, tflops_per_gpu, num_tokens_in_b, model_name
        )
    else:
        max_training_days = _estimate_training_time(
            model_size_in_b, num_tokens_in_b, gpu_count, tflops_per_gpu, model_name
        )

    print(
        f"You can train a {model_size_in_b}B parameter model in "
        f"{max_training_days} days using {gpu_count} GPUs. This result assumes "
        f"you are training to {num_tokens_in_b}B tokens, and each GPU achieves "
        f"{tflops_per_gpu} TFLOPS."
    )
    return model_size_in_b


def _estimate_model_size(max_training_days, gpu_count, tflops_per_gpu, num_tokens_in_b, model_name):
    """Estimates model size given time and hardware constraints.

    Arguments:
        max_training_days: float, number of days to train the model for.
        gpu_count: int, number of gpus to use (num_nodes * gpus_per_node).
        tflops_per_gpu: int, estimated number of TFLOPS/s per GPU.
        num_tokens_in_b: int, number of tokens to train the model for.
        model_name: str, name of the model, such as gpt3, t5, mt5...
    Output:
        model_size_in_b: int, number of parameters to use for training.
    """
    try:
        if model_name in ["gpt3", "t5"]:
            return round(
                (max_training_days * 3600 * 24 * gpu_count * tflops_per_gpu * 1e12)
                / (8 * num_tokens_in_b * 1e9) / 1e9,
                2,
            )
        else:
            raise NotImplementedError
    except Exception:
        print("Input values were not valid.")


def _estimate_training_time(
    model_size_in_b, gpu_count, tflops_per_gpu, num_tokens_in_b, model_name
):
    """Estimates training time for a given model size and hardware constraint.

    Arguments:
        model_size_in_b: int, number of parameters to use for training.
        gpu_count: int, number of gpus to use (num_nodes * gpus_per_node).
        tflops_per_gpu: int, estimated number of TFLOPS/s per GPU.
        num_tokens_in_b: int, number of tokens to train the model for.
        model_name: str, name of the model, such as gpt3, t5, mt5...
    Output:
        max_training_days: float, number of days it will take to train the model.
    """
    assert isinstance(model_size_in_b, float) or isinstance(
        model_size_in_b, int
    ), "model_size_in_b must be int or float."
    assert model_size_in_b > 0, "model_size_in_b must be larger than zero."

    try:
        if model_name in ["gpt3", "t5"]:
            max_training_days = round(
                (model_size_in_b * 1e9 * 8 * num_tokens_in_b * 1e9)
                / (3600 * 24 * gpu_count * tflops_per_gpu * 1e12),
                2,
            )
        else:
            raise NotImplementedError
    except Exception:
        print("Input values were not valid.")
    return max_training_days


def _calculate_gbs_tp_pp(model_size_in_b, model_name="gpt3"):
    """
    Calculates Global Batch Size (GBS), Tensor Parallelism (TP), and Pipeline
    Parallelism (PP) values, given a model size and model name.

    Arguments:
        model_size_in_b: float, the number of parameters in the model.
        model_name: str, name of the model, such as gpt3, t5, mt5...
    Output:
        gbs: int, global batch size to use for training.
        tp: int, tensor parallelism to use for training.
        pp: int, pipeline parallelism to use for training.
    """
    if model_name == "gpt3":
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
        elif model_size_in_b <= 1100:
            gbs, tp, pp = 2048, 8, 128
        else:
            print("No GPT-3 model larger than 1.1T parameters is supported.")
            raise ValueError
    elif model_name == "t5":
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
        elif model_size_in_b <= 42.0:
            gbs, tp, pp = 1920, 8, 4
        else:
            print("No T5 model larger than 42B parameters is supported.")
            raise ValueError
    else:
        raise NotImplementedError

    return gbs, tp, pp


def generate_base_config(
    model_size_in_b,
    nodes,
    gpus_per_node,
    max_training_days,
    num_tokens_in_b,
    vocab_size,
    model_name,
    cfg,
):
    """Generates base config for a given model name and size.

    Arguments:
        model_size_in_b: float, number of parameters in the model, if known.
        nodes: int, number of nodes to use for training.
        gpus_per_node: int, number of GPUs available in each node.
        max_training_days: float, number of days to train the model for.
        num_tokens_in_b: int, number of tokens to train the model for.
        model_name: str, name of the model, such as gpt3, t5, mt5...
        cfg: OmegaConf, full config object.
    Output:
        base_cfg: dict, base config object for the
    """
    # GBS: global batch size
    gbs, tp, pp = _calculate_gbs_tp_pp(model_size_in_b=model_size_in_b, model_name=model_name)

    base_cfg = utils.generic_base_config(cfg, model_name=model_name)

    # RUN
    base_cfg["run"]["name"] = f"{model_name}_{model_size_in_b}b"
    base_cfg["run"]["results_dir"] = "${base_results_dir}/${.name}"
    base_cfg["run"]["time_limit"] = (
        f"{int(max_training_days)}-"
        f"{int(24 * (max_training_days - int(max_training_days)))}:00:00"
    )

    # TRAINER
    base_cfg["trainer"]["num_nodes"] = nodes
    base_cfg["trainer"]["precision"] = "bf16"
    seq_length = base_cfg["model"]["data"]["seq_length"]
    base_cfg["trainer"]["max_steps"] = int((num_tokens_in_b * 1e9) / (seq_length * gbs))
    base_cfg["trainer"]["max_time"] = (
        f"{int(max_training_days)}:"
        f"{int(24 * (max_training_days - int(max_training_days))) - 1}:30:00"
    )

    # EXP_MANAGER
    wandb_cfg = cfg.get("wandb")
    enable = wandb_cfg.get("enable")
    project = wandb_cfg.get("project")
    if enable:
        base_cfg["exp_manager"]["create_wandb_logger"] = bool(enable)
        base_cfg["exp_manager"]["wandb_logger_kwargs"]["project"] = project

    # MODEL
    layers, hs, att_h, ffn, kv, lr = utils.calculate_model_size_params(model_size_in_b=model_size_in_b, vocab_size=vocab_size, seq_length=seq_length, model_name=model_name)
    base_cfg["model"]["num_layers"] = int(layers)
    base_cfg["model"]["global_batch_size"] = int(gbs)
    base_cfg["model"]["hidden_size"] = int(hs)
    base_cfg["model"]["num_attention_heads"] = int(att_h)
    if ffn is not None:
        base_cfg["model"]["ffn_hidden_size"] = int(ffn)
    if kv is not None:
        base_cfg["model"]["kv_channels"] = int(kv)
    base_cfg["model"]["init_method_std"] = round(0.64 / math.sqrt(hs), 6) if model_name == "gpt3" else 0.015
    base_cfg["model"]["optim"]["lr"] = lr
    base_cfg["model"]["optim"]["sched"]["min_lr"] = round(lr * 0.1, 8)
    if model_name == "gpt3":
        base_cfg["model"]["optim"]["sched"]["warmup_steps"] = int(
            0.0015 * base_cfg["trainer"]["max_steps"]
        )
        base_cfg["model"]["optim"]["sched"]["constant_steps"] = int(
            0.166 * base_cfg["trainer"]["max_steps"]
        )
    else:
        base_cfg["model"]["optim"]["sched"]["warmup_ratio"] = 0.01


    with open(f"{cfg.search_config.train_settings.logs}/base_cfg_{model_size_in_b}b.yaml", "w") as f:
        yaml.dump(base_cfg, f)
    return base_cfg
