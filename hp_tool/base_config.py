import os
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
            / (8 * num_tokens_in_b),
            2,
        )
    else:
        assert isinstance(model_size_in_b, float) or isinstance(
            model_size_in_b, int
        ), "model_size_in_b must be int or float."
        assert model_size_in_b > 0, "model_size_in_b must be larger than zero."
        max_training_days = round(
            (model_size_in_b * 8 * num_tokens_in_b)
            / (3600 * 24 * gpu_count * tflops_per_gpu * 1e12),
            2,
        )

    print(
        f"You can train a {model_size_in_b} parameter model in "
        f"{max_training_days} days using {gpu_count} GPUs. This result assumes"
        f"you are training to {num_tokens_in_b}B tokens, and each GPU achieves"
        f"{tflops_per_gpu} TFLOPS."
    )
    time.sleep(3)
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
    base_cfg["trainer"]["precision"] = 16 if model_size_in_b <= 5.5 else "bf16"
    mbs = base_cfg["model"]["micro_batch_size"]
    seq_length = base_cfg["model"]["data"]["seq_length"]
    base_cfg["trainer"]["max_steps"] = int((num_tokens_in_b * 1e9) / (seq_length * gbs))
    base_cfg["trainer"]["max_time"] = (
        f"{int(max_training_days)}:"
        f"{int(24 * (max_training_days - int(max_training_days))) - 1}:40:00"
    )

    # MODEL
    layers, hs, att_heads, lr = utils.calculate_layers_hs_lr(model_size)
    base_cfg["model"]["num_layers"] = int(layers)
    base_cfg["model"]["hidden_size"] = int(hs)
    base_cfg["model"]["num_attention_heads"] = int(att_heads)
    base_cfg["model"]["init_method_std"] = round(0.64 / math.sqrt(hidden_size), 6)
    base_cfg["model"]["optim"]["lr"] = lr
    base_cfg["model"]["optim"]["sched"]["min_lr"] = lr / 10
    base_cfg["model"]["optim"]["sched"]["warmup_steps"] = int(
        0.0015 * base_cfg["trainer"]["max_steps"]
    )
    base_cfg["model"]["optim"]["sched"]["constant_steps"] = int(
        0.166 * base_cfg["trainer"]["max_steps"]
    )

    with open(f"{cfg.base_results_dir}/base_cfg_{model_size}b.yaml", "w") as f:
        yaml.dump(base_cfg, f)

    return base_cfg


def calculate_tp_pp_mbs_grid(model_size_in_b):
    tp = [1, 2, 4, 8]
    pp = [1]
    mbs = [1, 2, 4, 8]
    if 4.0 < model_size_in_b <= 8.0:
        tp = [2, 4, 8]
    elif 8.0 < model_size_in_b <= 13.0:
        tp = [4, 8]
    elif 13.0 < model_size_in_b <= 23.0:
        tp = [8]
        pp = [1, 2]
    elif 23.0 < model_size_in_b <= 45.0:
        tp = [8]
        pp = [2, 4]
    elif 45.0 < model_size_in_b <= 95:
        tp = [8]
        pp = [4, 6, 8]
        mbs = [1, 2, 4]
    elif 95.0 < model_size_in_b <= 130.0:
        tp = [8]
        pp = [6, 8, 10, 12, 16]
        mbs = [1, 2, 4]
    elif 130.0 < model_size_in_b <= 195.0:
        tp = [8]
        pp = [8, 10, 12, 16, 20]
        mbs = [1, 2]
    elif 195.0 < model_size_in_b <= 395.0:
        tp = [8]
        pp = [16, 20, 24, 32, 40, 50]
        mbs = [1, 2]
    elif 395.0 < model_size_in_b <= 790.0:
        tp = [8]
        pp = [24, 32, 40, 50, 64]
        mbs = [1, 2]
    elif 790.0 < model_size_in_b <= 1100.0:
        tp = [8]
        pp = [32, 40, 50, 64, 72, 80]
        mbs = [1, 2]
    return tp, pp, mbs


def generate_grid_search_configs(base_cfg, model_size_in_b, cfg):
    tp_list, pp_list, mbs_list = calculate_tp_pp_grid(model_size_in_b=model_size_in_b)

    num_layers = base_cfg["model"]["num_layers"]

    results_cfgs = [[] for _ in range(num_layers + 1)]

    base_dir = f"{cfg.search_config.candidate_configs}/{model_size}b"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    max_minutes = cfg.search_train_config.train_settings.max_minutes_per_run
    # Generate Grid Search configs.
    for tp in tp_list:
        for pp in pp_list:
            act_ckpt_layers = [x for x in range(num_layers//pp + 1)]
            for act in act_ckpt_layers:
                for mbs in mbs_list:
                    new_cfg = utils.modify_cfg(base_cfg, act, tp, pp, mbs)
                    if new_cfg:  # Save candidate cfg.
                        file_name = f"tp_{tp}_pp_{pp}_mbs_{mbs}_act_ckpt_{act}.yaml"
                        first = cfg["search_train_config"]["slurm"]["job_name"].split(":")[
                            0
                        ]
                        second = new_cfg["slurm"]["job_name"].split(":")[-1]
                        new_cfg["slurm"]["job_name"] = f"{first}:{second}"
                        if new_cfg["slurm"]["gpus_per_task"] == "null":
                            del new_cfg["slurm"]["gpus_per_task"]
                        results_cfgs[act].append(file_name)
                        with open(f"{base_dir}/{file_name}", "w") as f:
                            yaml.dump(new_cfg, f)
    print("\nAll candidate configurations created correctly.\n")
    return base_dir, results_cfgs


def launch_grid_search_configs(base_dir, results_cfgs, cfg):
    limit = cfg["search_train_config"]["settings"]["limit_search_runs"]
    count = 0
    job_ids = []
    for cfg_list in results_cfgs:
        for config in cfg_list:
            conf = omegaconf.OmegaConf.load(f"{base_dir}/{config}")
            new_cfg = cfg.copy()
            new_cfg.training = conf
            hydra_args = convert_to_cli(new_cfg)
            job_id = run_training(new_cfg, hydra_args)
            job_ids.append(job_id[:-1])
            count += 1
            if count == limit:
                return job_ids
    return job_ids


def run_training(new_cfg, hydra_args):
    # TODO: Call from container.
    pass
    return job_id


def launch_throughput_measure(dependency_list, model_size, cfg):
    # Read config
    bignlp_path = cfg.get("bignlp_path")
    container_mounts = cfg.get("container_mounts")
    container = cfg.get("container")
    hp_cfg = cfg.get("search_train_config")

    # SLURM parameters
    slurm_cfg = hp_cfg.get("slurm")
    partition = slurm_cfg.get("partition")
    account = slurm_cfg.get("account")
    time_limit = "30:00"
    nodes = 1
    exclusive = slurm_cfg.get("exclusive")
    mem = slurm_cfg.get("mem")
    overcommit = slurm_cfg.get("overcommit")
    ntasks_per_node = 1
    gpus_per_task = None
    dependency = None
    if dependency_list is not None and len(dependency_list) > 0:
        dependency = ":".join(dependency_list)
    job_name = slurm_cfg.get("job_name")

    # Settings parameters
    settings = hp_cfg.get("settings")
    final_log_dir = settings.final_result_logs
    if not os.path.exists(final_log_dir):
        os.makedirs(final_log_dir)

    # Process container-mounts.
    mounts_str = f"{bignlp_path}:{bignlp_path}"
    if container_mounts is not None:
        assert isinstance(
            container_mounts, omegaconf.listconfig.ListConfig
        ), "container_mounts must be a list."
        for mount in container_mounts:
            if mount is not None and isinstance(mount, str):
                mounts_str += f",{mount}:{mount}"

    flags = (
        f"--container-image {container} "
        f"--container-mounts {mounts_str} "
        f"-o {final_log_dir}/compare_throughput_{model_size}b-%j.log "
        f"-e {final_log_dir}/compare_throughput_{model_size}b-%j.error "
    )
    new_script_path = os.path.join(
        bignlp_path, "search_train_config/compare_throughput.sh"
    )
    code_path = os.path.join(
        bignlp_path, "search_train_config/compare_throughput_results.py"
    )
    train_cmd = f"HYDRA_FULL_ERROR=1 python3 -u {code_path} search_train_config.settings.model_size_in_b={model_size}"
    utils.create_slurm_file(
        new_script_path=new_script_path,
        train_cmd=train_cmd,
        job_name=job_name,
        flags=flags,
        dependency=dependency,
        exclusive=exclusive,
        mem=mem,
        overcommit=overcommit,
        time=time_limit,
        nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        gpus_per_task=gpus_per_task,
        partition=partition,
        account=account,
    )

    job_id = subprocess.check_output(
        [f"sbatch --parsable {new_script_path}"], shell=True
    )
    dependency = job_id.decode("utf-8")
    print(f"Submitted job to select optimal throughput with job id: {dependency}")
    return dependency
