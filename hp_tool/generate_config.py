import os
import math

import subprocess
import submitit
import yaml
import omegaconf

from search_train_config import utils
from train_scripts.train import run_training
from main import convert_to_cli


def calculate_model_size(gpu_count, max_training_days, model_size):
    if model_size is None:
        model_size = utils.model_size_from_constraints(gpu_count, max_training_days)
    return model_size


def generate_config_for_model_size(model_size, nodes, gpus_per_node, max_training_days, cfg):
    # GBS: global batch size
    if model_size <= 1.0:
        gbs = 256
        tp = 1
    elif model_size <= 4.0:
        gbs = 720
        tp = 1
    elif model_size <= 8.0:
        gbs = 1440
        tp = 2
    elif model_size <= 13.0:
        gbs = 1440
        tp = 4
    elif model_size <= 20.8:
        gbs = 1440
        tp = 8
    else:
        print("No model larger than 20B parameters is supported.")

    base_cfg = utils.generic_base_config(cfg.search_train_config)

    # SLURM
    base_cfg["slurm"]["nodes"] = int(nodes)
    base_cfg["slurm"]["ntasks_per_node"] = int(gpus_per_node)
    base_cfg["slurm"]["job_name"] = f"bignlp-gpt3:{model_size}b"
    base_cfg["slurm"]["time_limit"] = f"{int(max_training_days)}-{int(24 * (max_training_days - int(max_training_days)))}:00:00"

    # RUN
    base_cfg["run"]["name"] = f"{model_size}b"
    base_cfg["run"]["log_dir"] = f"${{bignlp_path}}/search_train_config/candidate_logs/{model_size}b"

    # TRAINER
    if model_size <= 5.5:
        base_cfg["trainer"]["precision"] = 16
        base_cfg["model"]["fused_fp16"] = True
        base_cfg["model"]["fused_bf16"] = False
    else:
        base_cfg["trainer"]["precision"] = "bf16"
        base_cfg["model"]["fused_fp16"] = False
        base_cfg["model"]["fused_bf16"] = True
    mbs = base_cfg["model"]["micro_batch_size"]
    seq_length = base_cfg["model"]["data"]["seq_length"]
    base_cfg["trainer"]["accumulate_grad_batches"] = int(gbs // (mbs * nodes * gpus_per_node / tp)) # 360 / (8 * 8 / 1)
    accumulate_grad_batches = base_cfg["trainer"]["accumulate_grad_batches"]
    base_cfg["trainer"]["max_steps"] = int((3e11 / seq_length) // (mbs * nodes * gpus_per_node * accumulate_grad_batches / tp))
    base_cfg["trainer"]["max_time"] = f"{int(max_training_days)}:{int(24 * (max_training_days - int(max_training_days))) - 1}:40:00"

    # MODEL
    num_layers, hidden_size, att_heads, lr = utils.calculate_num_layers_hidden_size_learning_rate(model_size)
    base_cfg["model"]["num_layers"] = int(num_layers)
    base_cfg["model"]["hidden_size"] = int(hidden_size)
    base_cfg["model"]["num_attention_heads"] = int(att_heads)
    base_cfg["model"]["init_method_std"] = round(0.64 / math.sqrt(hidden_size), 6)
    base_cfg["model"]["optim"]["lr"] = lr
    base_cfg["model"]["optim"]["sched"]["min_lr"] = lr / 10
    base_cfg["model"]["optim"]["sched"]["warmup_steps"] = int(0.0015 * base_cfg["trainer"]["max_steps"])
    base_cfg["model"]["optim"]["sched"]["constant_steps"] = int(0.166 * base_cfg["trainer"]["max_steps"])

    with open(f"{cfg.bignlp_path}/search_train_config/base_cfg_{model_size}b.yaml", "w") as f:
        yaml.dump(base_cfg, f)

    return base_cfg, gbs


def generate_grid_search_configs(base_cfg, gbs, model_size, cfg):
    if model_size <= 1.0:
        tensor_parallel = [1,2,4,8]
    elif model_size <= 4.0:
        tensor_parallel = [1,2,4,8]
    elif model_size <= 8.0:
        tensor_parallel = [2,4,8]
    elif model_size <= 13.0:
        tensor_parallel = [4,8]
    else:
        tensor_parallel = [8]
    micro_batch_size = [1,2,4,8]
    num_layers = base_cfg["model"]["num_layers"]
    act_ckpt_layers = [x for x in range(num_layers+1)]

    results_cfgs = [[] for _ in range(num_layers+1)]

    base_dir = f"{cfg.bignlp_path}/search_train_config/candidate_configs/{model_size}b"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    max_mins = cfg.search_train_config.settings.max_mins_per_run
    # Generate Grid Search configs.
    for act_layers in act_ckpt_layers:
        for tp in tensor_parallel:
            for mbs in micro_batch_size:
                new_cfg = utils.modify_cfg(base_cfg, gbs, act_layers, tp, mbs, max_mins, model_size)
                if new_cfg:  # Save candidate cfg.
                    file_name = f"tp_{tp}_mbs_{mbs}_act_ckpt_{act_layers}.yaml"
                    first = cfg['search_train_config']['slurm']['job_name'].split(':')[0]
                    second = new_cfg["slurm"]["job_name"].split(':')[-1]
                    new_cfg["slurm"]["job_name"] = f"{first}:{second}"
                    if new_cfg["slurm"]["gpus_per_task"] == "null":
                        del new_cfg["slurm"]["gpus_per_task"]
                    results_cfgs[act_layers].append(file_name)
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
        assert isinstance(container_mounts, omegaconf.listconfig.ListConfig), "container_mounts must be a list."
        for mount in container_mounts:
            if mount is not None and isinstance(mount, str):
                mounts_str += f",{mount}:{mount}"

    flags = (
        f"--container-image {container} "
        f"--container-mounts {mounts_str} "
        f"-o {final_log_dir}/compare_throughput_{model_size}b-%j.log "
        f"-e {final_log_dir}/compare_throughput_{model_size}b-%j.error "
    )
    new_script_path = os.path.join(bignlp_path, "search_train_config/compare_throughput.sh")
    code_path = os.path.join(bignlp_path, "search_train_config/compare_throughput_results.py")
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

