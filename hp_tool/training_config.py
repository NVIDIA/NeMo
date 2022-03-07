import os
import yaml
import subprocess

import omegaconf
from omegaconf import OmegaConf

from hp_tool import utils, train


def search_training_config(base_cfg, model_size, cfg):
    # Generate candidate configs.
    base_dir, results_cfgs = generate_grid_search_configs(
        base_cfg, model_size, cfg
    )
    # Launch candidate configs.
    #job_ids = launch_grid_search_configs(base_dir, results_cfgs, cfg)
    job_ids = None
    # Measure and compare throughputs for each config.
    launch_throughput_measure(job_ids, model_size, cfg)


def generate_grid_search_configs(base_cfg, model_size_in_b, cfg):
    num_layers = base_cfg["model"]["num_layers"]
    results_cfgs = [[] for _ in range(num_layers + 1)]
    
    tp_list, pp_list, mbs_list = _calculate_tp_pp_mbs_grid(model_size_in_b=model_size_in_b, num_layers=num_layers)

    base_dir = f"{cfg.search_config.train_settings.candidate_configs}/{model_size_in_b}b"
    os.makedirs(base_dir, exist_ok=True)

    max_minutes = cfg.search_config.train_settings.max_minutes_per_run
    
    valid_pp_list = []
    for tp in tp_list:
        for pp in pp_list:
            act_ckpt_layers = [x for x in range(num_layers//pp + 1)]
            for act in act_ckpt_layers:
                for mbs in mbs_list:
                    num_gpus = base_cfg["trainer"]["num_nodes"] * base_cfg["trainer"]["gpus"]
                    gbs = base_cfg["model"]["global_batch_size"]
                    att_heads = base_cfg["model"]["num_attention_heads"]
                    num_layers = base_cfg["model"]["num_layers"]
                    mod_gbs = gbs % (mbs * num_gpus / (tp * pp))
                    mod_att_heads = att_heads % tp
                    mod_layers = num_layers % pp
                    if mod_gbs == 0 and mod_att_heads == 0 and mod_layers == 0:
                        valid_pp_list.append(pp)

    # Generate grid search configs.
    for tp in tp_list:
        for pp in pp_list:
            act_ckpt_layers = [x for x in range(num_layers//pp + 1)]
            for act in act_ckpt_layers:
                for mbs in mbs_list:
                    new_cfg = utils.modify_cfg(base_cfg, act, tp, pp, mbs, max_minutes, max(valid_pp_list))
                    if new_cfg:  # Save candidate cfg.
                        file_name = f"{model_size_in_b}b_tp_{tp}_pp_{pp}_mbs_{mbs}_act_ckpt_{act}.yaml"
                        results_cfgs[act].append(file_name)
                        with open(f"{base_dir}/{file_name}", "w") as f:
                            yaml.dump(new_cfg, f)
    print("\nAll candidate configurations created correctly.\n")
    return base_dir, results_cfgs


def _calculate_tp_pp_mbs_grid(model_size_in_b, num_layers):
    valid_pp = [x for x in range(1, num_layers+1) if num_layers % x == 0] # Only divisors of num_layers are possible.
    tp = [1, 2, 4, 8]
    pp = [1]
    mbs = [1, 2, 4, 8]
    if 4.0 < model_size_in_b <= 8.0:
        tp = [2, 4, 8]
    elif 8.0 < model_size_in_b <= 13.0:
        tp = [4, 8]
    elif 13.0 < model_size_in_b <= 23.0:
        tp = [8]
        pp = [x for x in valid_pp if x < 6]
    elif 23.0 < model_size_in_b <= 45.0:
        tp = [8]
        pp = [x for x in valid_pp if 1 < x < 7]
    elif 45.0 < model_size_in_b <= 95:
        tp = [8]
        pp = [x for x in valid_pp if 3 < x < 11]
        mbs = [1, 2, 4]
    elif 95.0 < model_size_in_b <= 130.0:
        tp = [8]
        pp = [x for x in valid_pp if 5 < x < 21]
        mbs = [1, 2, 4]
    elif 130.0 < model_size_in_b <= 195.0:
        tp = [8]
        pp = [x for x in valid_pp if 7 < x < 29]
        mbs = [1, 2]
    elif 195.0 < model_size_in_b <= 395.0:
        tp = [8]
        pp = [x for x in valid_pp if 15 < x < 65]
        mbs = [1, 2]
    elif 395.0 < model_size_in_b <= 790.0:
        tp = [8]
        pp = [x for x in valid_pp if 23 < x < 71]
        mbs = [1, 2]
    elif 790.0 < model_size_in_b <= 1100.0:
        tp = [8]
        pp = [x for x in valid_pp if 29 < x < 131]
        mbs = [1, 2]
    return tp, pp, mbs


def launch_grid_search_configs(base_dir, results_cfgs, cfg):
    limit = cfg.search_config.train_settings.limit_search_runs
    count = 0
    job_ids = []
    for cfg_list in results_cfgs:
        for config in cfg_list:
            conf = OmegaConf.load(f"{base_dir}/{config}")
            new_cfg = create_bignlp_config(cfg)
            new_cfg.training = conf
            new_cfg.cluster = cfg.cluster
            job_id = train.run_training(new_cfg, cfg.bignlp_hp_tool_path)
            if job_id is not None:
                job_ids.append(job_id[:-1])
            count += 1
            if count == limit:
                return job_ids
    return job_ids


def create_bignlp_config(cfg):
    results_dir = cfg.search_config.train_settings.candidate_logs
    training_container = cfg.training_container
    data_dir = cfg.data_dir
    model_size = cfg.search_config.train_settings.model_size_in_b

    s = f"""
    training: null
    cluster: null

    run_data_preparation: False
    run_training: True
    run_conversion: False
    run_finetuning: False
    run_evaluation: False

    cluster_type: bcm
    training_config: gpt3/5b
    bignlp_path: /opt/bignlp/bignlp-scripts
    data_dir: {data_dir}
    base_results_dir: {results_dir}/{model_size}b
    container_mounts:
      - {results_dir}:/opt/bignlp/bignlp-scripts/results
    container: {training_container}
    """
    new_cfg = OmegaConf.create(s)
    return new_cfg


def launch_throughput_measure(dependency_list, model_size, cfg):
    # Read config
    bignlp_hp_tool_path = cfg.get("bignlp_hp_tool_path")
    container_mounts = cfg.container_mounts
    container = cfg.training_container
    hp_cfg = cfg.search_config

    # CLUSTER parameters
    cluster_cfg = cfg.get("cluster")
    partition = cluster_cfg.get("partition")
    account = cluster_cfg.account
    time_limit = "30:00"
    nodes = 1
    exclusive = cluster_cfg.exclusive
    mem = cluster_cfg.mem
    overcommit = cluster_cfg.overcommit
    ntasks_per_node = 1
    gpus_per_task = None
    dependency = None
    if dependency_list is not None and len(dependency_list) > 0:
        dependency = ":".join(dependency_list)
    job_name = f"{cluster_cfg.job_name_prefix}latency_measure"

    # Settings parameters
    train_settings = hp_cfg.train_settings
    final_log_dir = train_settings.final_result_logs
    if not os.path.exists(final_log_dir):
        os.makedirs(final_log_dir)

    # Process container-mounts.
    mounts_str = f"{bignlp_hp_tool_path}:{bignlp_hp_tool_path}"
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
        bignlp_hp_tool_path, "hp_tool/scripts/compare_throughput.sh"
    )
    code_path = os.path.join(
        bignlp_hp_tool_path, "hp_tool/scripts/compare_throughput_results.py"
    )
    train_cmd = f"HYDRA_FULL_ERROR=1 python3 -u {code_path} search_config.train_settings.model_size_in_b={model_size}"
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

