import os
import shutil
import math
import yaml
import subprocess

import omegaconf
from omegaconf import OmegaConf

from hp_tool import utils, train


def search_training_config(base_cfg, model_size, model_name, cfg):
    # Generate candidate configs.
    base_dir, results_cfgs, num_nodes = generate_grid_search_configs(base_cfg, model_size, model_name, cfg)
    # Launch candidate configs.
    job_ids = launch_grid_search_configs(base_dir, results_cfgs, model_name, cfg)
    # Measure and compare throughputs for each config.
    launch_throughput_measure(job_ids, model_name, model_size, num_nodes, cfg)


def generate_grid_search_configs(base_cfg, model_size_in_b, model_name, cfg):
    search_cfg = cfg.get("search_config")
    train_cfg = search_cfg.get("train_settings")
    gpus_per_node = train_cfg.get("gpus_per_node")
    num_nodes = train_cfg.get("num_nodes")
    act_layers = train_cfg.get("act_ckpt_layers")

    # 2 * num_layers is needed because of encoder/decoder architecture.
    multiplier = 1 if model_name == "gpt3" else 2

    num_layers = base_cfg["model"]["num_layers"] if model_name == "gpt3" else base_cfg["model"]["encoder"]["num_layers"]
    act_granularity = base_cfg["model"].get("activations_checkpoint_granularity", "full")

    tp_list, pp_list, mbs_list = _calculate_tp_pp_mbs_grid(
        model_size_in_b=model_size_in_b,
        num_layers=num_layers,
        model_name=model_name,
        train_cfg=train_cfg,
    )

    base_dir = f"{cfg.search_config.train_settings.logs}/candidate_configs"
    os.makedirs(base_dir, exist_ok=True)

    max_minutes = train_cfg.get("max_minutes_per_run")
    max_steps = train_cfg.get("max_steps_per_run")

    act_multiple = 1
    if act_granularity == "full":
        results_cfgs = [[] for _ in range(multiplier * num_layers + 1)]
        if model_name != "gpt3":
            if model_size_in_b < 1.0:
                act_multiple = 2
            elif 1.0 <= model_size_in_b < 26.0:
                act_multiple = 4
            else:
                act_multiple = 8
    else:
        results_cfgs = [[]]

    valid_tp_pp_list = []
    for tp in tp_list:
        for pp in pp_list:
            for mbs in mbs_list:
                num_gpus = base_cfg["trainer"]["num_nodes"] * base_cfg["trainer"]["devices"]
                gbs = base_cfg["model"]["global_batch_size"]
                if model_name == "gpt3":
                    att_heads = base_cfg["model"]["num_attention_heads"]
                    num_layers = base_cfg["model"]["num_layers"]
                else:
                    att_heads = base_cfg["model"]["encoder"]["num_attention_heads"]
                    num_layers = base_cfg["model"]["encoder"]["num_layers"]
                mod_gbs = gbs % (mbs * num_gpus / (tp * pp))
                mod_att_heads = att_heads % tp
                mod_layers = (multiplier * num_layers) % pp
                if mod_gbs == 0 and mod_att_heads == 0 and mod_layers == 0:
                    valid_tp_pp_list.append((tp, pp))

    # Generate grid search configs.
    for tp in tp_list:
        for pp in pp_list:
            for mbs in mbs_list:
                if act_granularity == "full":
                    act_ckpt_layers = [
                        x for x in range(multiplier * num_layers // pp + 1) if x % act_multiple == 0
                    ]
                    if act_layers is not None and act_layers != "auto":
                        act_ckpt_layers = act_layers
                    for act in act_ckpt_layers:
                        new_cfg = utils.modify_cfg(
                            base_cfg=base_cfg,
                            act=act,
                            tp=tp,
                            pp=pp,
                            mbs=mbs,
                            max_minutes=max_minutes,
                            max_steps=max_steps,
                            num_nodes=num_nodes,
                            model_name=model_name,
                        )
                        if new_cfg:  # Save candidate cfg.
                            file_name = f"{model_name}_{model_size_in_b}b_{num_nodes}nodes_tp_{tp}_pp_{pp}_mbs_{mbs}_act_ckpt_{act}.yaml"
                            results_cfgs[act].append(file_name)
                            with open(f"{base_dir}/{file_name}", "w") as f:
                                yaml.dump(new_cfg, f)
                else:
                    new_cfg = utils.modify_cfg(
                        base_cfg=base_cfg,
                        act=None,
                        tp=tp,
                        pp=pp,
                        mbs=mbs,
                        max_minutes=max_minutes,
                        max_steps=max_steps,
                        num_nodes=num_nodes,
                    )
                    if new_cfg:  # Save candidate cfg.
                        file_name = f"{model_name}_{model_size_in_b}b_{num_nodes}nodes_tp_{tp}_pp_{pp}_mbs_{mbs}_act_ckpt_selective.yaml"
                        results_cfgs[0].append(file_name)
                        with open(f"{base_dir}/{file_name}", "w") as f:
                            yaml.dump(new_cfg, f)

    print("\nAll candidate configurations created correctly.\n")
    return base_dir, results_cfgs, num_nodes
            
def _tp_pp_mbs_grid_gpt3_80gb(model_size_in_b, valid_pp):
    """Selects grid search space for TP, PP, MBS parameters for GPT-3 and 80GB GPUs.
    """
    tp = [1, 2, 4, 8]
    pp = [1]
    mbs = [1, 2, 3, 4, 6, 8]
    if model_size_in_b <= 1.0:
        tp = [1, 2]
    elif 1.0 < model_size_in_b <= 4.0:
        tp = [1, 2, 4]
    elif 4.0 < model_size_in_b <= 8.0:
        tp = [1, 2, 4]
    elif 8.0 < model_size_in_b <= 13.0:
        tp = [1, 2, 4, 8]
    elif 13.0 < model_size_in_b <= 23.0:
        tp = [1, 2, 4, 8]
        pp = [x for x in valid_pp if x <= 4]
    elif 23.0 < model_size_in_b <= 45.0:
        tp = [2, 4, 8]
        pp = [x for x in valid_pp if 1 <= x <= 4]
        mbs = [1, 2, 4, 8]
    elif 45.0 < model_size_in_b <= 95:
        tp = [2, 4, 8]
        pp = [x for x in valid_pp if 1 <= x <= 8]
        mbs = [1, 2, 4, 8]
    elif 95.0 < model_size_in_b <= 130.0:
        tp = [2, 4, 8]
        pp = [x for x in valid_pp if 1 <= x <= 16]
        mbs = [1, 2, 4, 8]
    elif 130.0 < model_size_in_b <= 195.0:
        tp = [4, 8]
        pp = [x for x in valid_pp if 2 <= x <= 16]
        mbs = [1, 2, 4]
    elif 195.0 < model_size_in_b <= 395.0:
        tp = [4, 8]
        pp = [x for x in valid_pp if 2 <= x <= 32]
        mbs = [1, 2, 4]
    elif 395.0 < model_size_in_b <= 790.0:
        tp = [4, 8]
        pp = [x for x in valid_pp if 4 <= x <= 100]
        mbs = [1, 2, 4]
    elif 790.0 < model_size_in_b <= 1100.0:
        tp = [4, 8]
        pp = [x for x in valid_pp if 4 <= x <= 130]
        mbs = [1, 2, 4]
    return tp, pp, mbs

def _tp_pp_mbs_grid_gpt3_40gb(model_size_in_b, valid_pp):
    """Selects grid search space for TP, PP, MBS parameters for GPT-3 and 40GB GPUs.
    """
    tp = [1, 2, 4, 8]
    pp = [1]
    mbs = [1, 2, 4, 6, 8, 10, 12, 16]
    if model_size_in_b <= 1.0:
        tp = [1, 2, 4]
        mbs = [1, 2, 4, 8]
    elif 1.0 < model_size_in_b <= 4.0:
        tp = [1, 2, 4, 8]
        mbs = [1, 2, 4, 8]
    elif 4.0 < model_size_in_b <= 8.0:
        tp = [4, 8]
        pp = [1, 2]
        mbs = [1, 2, 4]
    elif 8.0 < model_size_in_b <= 13.0:
        tp = [8]
        pp = [1, 2, 4]
        mbs = [1, 2, 4]
    elif 13.0 < model_size_in_b <= 23.0:
        tp = [2, 4, 8]
        pp = [x for x in valid_pp if 1 <= x <= 8]
    elif 23.0 < model_size_in_b <= 45.0:
        tp = [4, 8]
        pp = [x for x in valid_pp if 1 <= x <= 12]
        mbs = [1, 2, 4]
    elif 45.0 < model_size_in_b <= 95:
        tp = [4, 8]
        pp = [x for x in valid_pp if 1 <= x <= 16]
        mbs = [1, 2, 4]
    elif 95.0 < model_size_in_b <= 130.0:
        tp = [4, 8]
        pp = [x for x in valid_pp if 2 <= x <= 26]
        mbs = [1, 2]
    elif 130.0 < model_size_in_b <= 195.0:
        tp = [4, 8]
        pp = [x for x in valid_pp if 2 <= x <= 32]
        mbs = [1, 2]
    elif 195.0 < model_size_in_b <= 395.0:
        tp = [4, 8]
        pp = [x for x in valid_pp if 4 <= x <= 64]
        mbs = [1, 2]
    elif 395.0 < model_size_in_b <= 790.0:
        tp = [4, 8]
        pp = [x for x in valid_pp if 8 <= x <= 128]
        mbs = [1, 2]
    elif 790.0 < model_size_in_b <= 1100.0:
        tp = [4, 8]
        pp = [x for x in valid_pp if 8 <= x <= 192]
        mbs = [1, 2]
    return tp, pp, mbs
    
def _tp_pp_mbs_grid_t5_80gb(model_size_in_b, valid_pp):
    """Selects grid search space for TP, PP, MBS parameters for T5/mT5 and 80GB GPUs.
    """
    tp = [1, 2, 4, 8]
    pp = [1]
    mbs = [1, 2, 4, 6, 8, 12, 16]
    if model_size_in_b <= 1.0:
        tp = [1, 2]
        mbs = [16, 32, 64, 128]
    elif 1.0 < model_size_in_b <= 4.0:
        tp = [1, 2, 4]
        mbs = [4, 6, 8, 12, 16, 24, 32, 48]
    elif 4.0 < model_size_in_b <= 8.0:
        tp = [2, 4, 8]
        mbs = [4, 6, 8, 12, 16, 24, 32]
    elif 8.0 < model_size_in_b <= 14.5:
        tp = [4, 8]
        mbs = [2, 4, 6, 8, 12, 16, 24]
    elif 14.5 < model_size_in_b <= 25.9:
        tp = [4, 8]
        pp = [x for x in valid_pp if 1 <= x <= 2]
        mbs = [1, 2, 4, 6, 8]
    elif 25.9 < model_size_in_b <= 43.0:
        tp = [4, 8]
        pp = [x for x in valid_pp if 1 <= x <= 4]
        mbs = [1, 2, 4, 6, 8]
    return tp, pp, mbs

def _tp_pp_mbs_grid_t5_40gb(model_size_in_b, valid_pp):
    """Selects grid search space for TP, PP, MBS parameters for T5/mT5 and 40GB GPUs.
    """
    tp = [1, 2, 4, 8]
    pp = [1]
    mbs = [1, 2, 4, 6, 8, 12, 16]
    if model_size_in_b <= 1.0:
        tp = [1, 2]
        mbs = [16, 32, 64, 128]
    elif 1.0 < model_size_in_b <= 4.0:
        tp = [1, 2, 4]
        mbs = [4, 8, 12, 16, 24, 32, 48]
    elif 4.0 < model_size_in_b <= 8.0:
        tp = [2, 4, 8]
        mbs = [4, 6, 8, 12, 16, 24]
    elif 8.0 < model_size_in_b <= 14.5:
        tp = [4, 8]
        pp = [x for x in valid_pp if 1 <= x <= 2]
        mbs = [2, 4, 6, 8, 12, 16]
    elif 14.5 < model_size_in_b <= 25.9:
        tp = [4, 8]
        pp = [x for x in valid_pp if 1 <= x <= 8]
        mbs = [1, 2, 4, 6, 8]
    elif 25.9 < model_size_in_b <= 43.0:
        tp = [4, 8]
        pp = [x for x in valid_pp if 1 <= x <= 8]
        mbs = [1, 2, 4, 6, 8]
    return tp, pp, mbs

def _calculate_tp_pp_mbs_grid(model_size_in_b, num_layers, model_name, train_cfg):
    tp_sizes = train_cfg.get("tensor_parallel_sizes")
    pp_sizes = train_cfg.get("pipeline_parallel_sizes")
    mbs_sizes = train_cfg.get("micro_batch_sizes")
    gpu_memory_gb = train_cfg.get("gpu_memory_gb")

    multiplier = 1 if model_name == "gpt3" else 2
    init_pp = [] if model_name == "gpt3" else [1]
    valid_pp = init_pp + [
        multiplier * x for x in range(1, num_layers + 1) if num_layers % x == 0
    ]  # Only divisors of num_layers are possible.

    if model_name == "gpt3":
        if gpu_memory_gb == 80:
            tp, pp, mbs = _tp_pp_mbs_grid_gpt3_80gb(model_size_in_b=model_size_in_b, valid_pp=valid_pp)
        elif gpu_memory_gb == 40:
            tp, pp, mbs = _tp_pp_mbs_grid_gpt3_40gb(model_size_in_b=model_size_in_b, valid_pp=valid_pp)
    elif model_name in ["t5", "mt5"]:
        if gpu_memory_gb == 80:
            tp, pp, mbs = _tp_pp_mbs_grid_t5_80gb(model_size_in_b=model_size_in_b, valid_pp=valid_pp)
        elif gpu_memory_gb == 40:
            tp, pp, mbs = _tp_pp_mbs_grid_t5_40gb(model_size_in_b=model_size_in_b, valid_pp=valid_pp)
    else:
        raise NotImplementedError("Model name not implemented.")

    # Override the tp, pp, mbs search if indicated in the config params.
    if tp_sizes is not None and tp_sizes != "auto":
        tp = tp_sizes
    if pp_sizes is not None and pp_sizes != "auto":
        pp = pp_sizes
    if mbs_sizes is not None and mbs_sizes != "auto":
        mbs = mbs_sizes
    return tp, pp, mbs


def launch_grid_search_configs(base_dir, results_cfgs, model_name, cfg):
    """Launches training jobs for the grid search in parallel. The limit of how many
    jobs to launch is specified by limit_search_runs.

    Arguments:
        base_dir: str, location where the configs are stored.
        results_cfgs: list, list of config names.
        cfg: OmegaConf, the general config object.
    Output
        job_ids: list, list of job ids for all the training jobs.
    """
    bignlp_hp_tool_path = cfg.get("bignlp_hp_tool_path")
    bignlp_scripts_path = cfg.get("bignlp_scripts_path")
    
    search_cfg = cfg.get("search_config")
    train_cfg = search_cfg.get("train_settings")
    limit = train_cfg.get("limit_search_runs")
    results_dir = os.path.join(train_cfg.get("logs"), "training_logs")

    job_ids = []
    for cfg_list in results_cfgs:
        for file_name in cfg_list:
            src_file = os.path.join(base_dir, file_name)
            dst_dir = os.path.join(bignlp_scripts_path, "conf/training", model_name, file_name)
            shutil.copyfile(src_file, dst_dir)
            job_id = train.run_training(file_name, bignlp_hp_tool_path, bignlp_scripts_path, model_name, results_dir, cfg)
            os.remove(dst_dir)
            
            if job_id is not None:
                job_ids.append(job_id[:-1])
            if len(job_ids) == limit:
                return job_ids
    return job_ids


def launch_throughput_measure(dependency_list, model_name, model_size_in_b, num_nodes, cfg):
    """Launch job that measures the throughput of each run in the grid search. This
    job will get scheduled with dependencies on all the job ids in dependency_list,
    so it will only start running once all the jobs are finished.

    Arguments:
        dependency_list: list, list of all the job_ids this job will depend on.
        model_name: str, name of the model, i.e. gpt3, t5, mt5.
        model_size_in_b: float, model size in billions of parameters.
        cfg: OmegaCOnf, general config object.
    Output:
        dependency: str, job_id of the current job.
    """
    # Read config
    bignlp_hp_tool_path = cfg.get("bignlp_hp_tool_path")
    container_mounts = cfg.get("container_mounts")
    container = cfg.get("training_container")
    hp_cfg = cfg.get("search_config")
    base_results_dir = cfg.get("base_results_dir")

    # CLUSTER parameters
    cluster_cfg = cfg.get("cluster")
    partition = cluster_cfg.get("partition")
    account = cluster_cfg.get("account")
    time_limit = "10:00"
    exclusive = cluster_cfg.get("exclusive")
    mem = cluster_cfg.get("mem")
    overcommit = cluster_cfg.get("overcommit")
    ntasks_per_node = 1
    gpus_per_task = None
    gpus_per_node = None
    dependency = None
    if dependency_list is not None and len(dependency_list) > 0:
        dependency = ":".join(dependency_list)
    job_name = f"{cluster_cfg.get('job_name_prefix')}latency_measure"

    # Settings parameters
    train_settings = hp_cfg.get("train_settings")
    log_dir = train_settings.get("logs")
    final_log_dir = os.path.join(log_dir, "final_result")
    os.makedirs(final_log_dir, exist_ok=True)

    # Process container-mounts.
    mounts_str = f"{bignlp_hp_tool_path}:{bignlp_hp_tool_path},{base_results_dir}:{base_results_dir}"
    mounts_str += utils.add_container_mounts(container_mounts)

    flags = (
        f"--container-image {container} "
        f"--container-mounts {mounts_str} "
        f"--no-container-mount-home "
    )
    if cfg.get("ci_test"):  # Whether this job is running in CI or not.
        flags += f"-o {log_dir}/slurm_%j.log "
    else:
        flags += (
            f"-o {final_log_dir}/compare_throughput_{model_size_in_b}b_{num_nodes}nodes-%j.log "
            f"-e {final_log_dir}/compare_throughput_{model_size_in_b}b_{num_nodes}nodes-%j.error "
        )

    new_script_path = os.path.join(bignlp_hp_tool_path, "hp_tool/scripts/compare_throughput.sh")
    code_path = os.path.join(bignlp_hp_tool_path, "hp_tool/scripts/compare_throughput_results.py")
    train_cmd = f"HYDRA_FULL_ERROR=1 python3 -u {code_path} search_config.train_settings.model_size_in_b={model_size_in_b} search_config={model_name}/{model_size_in_b}b search_config_value={model_name}/{model_size_in_b}b +nodes={num_nodes} "
    utils.create_slurm_file(
        new_script_path=new_script_path,
        cmds=[train_cmd],
        job_name=job_name,
        flags=flags,
        dependency=dependency,
        exclusive=exclusive,
        mem=mem,
        overcommit=overcommit,
        time=time_limit,
        nodes=1,
        ntasks_per_node=ntasks_per_node,
        gpus_per_task=gpus_per_task,
        gpus_per_node=gpus_per_node,
        partition=partition,
        account=account,
    )
    if cfg.get("ci_test"):
        job_id = subprocess.check_output([f'sbatch {new_script_path} | tee "{log_dir}/launcher.log" '], shell=True)
    else:
        job_id = subprocess.check_output([f"sbatch --parsable {new_script_path}"], shell=True)
    dependency = job_id.decode("utf-8")
    print(f"Submitted job to select optimal throughput with job id: {dependency}")
    return dependency
