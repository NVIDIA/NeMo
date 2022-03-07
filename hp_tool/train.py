import sys
import os
import subprocess

import hydra
import omegaconf

from hp_tool.utils import convert_to_cli


def create_slurm_file(
    new_script_path,
    train_cmd,
    job_name,
    flags="",
    dependency=None,
    time="04:00:00",
    exclusive=True,
    mem=0,
    overcommit=True,
    nodes=1,
    ntasks_per_node=8,
    gpus_per_task=1,
    partition="batch",
    account=None,
):
    """
    Creates a slurm file to launch a training job.
    """
    with open(new_script_path, "w") as f:
        f.writelines("#!/bin/bash\n")
        f.writelines(f"#SBATCH --nodes={nodes}\n")
        f.writelines(f"#SBATCH --ntasks-per-node={ntasks_per_node}\n")
        if gpus_per_task is not None:
            f.writelines(f"#SBATCH --gpus-per-task={gpus_per_task}\n")
        if dependency is not None:
            if dependency != "singleton":
                dependency = f"afterany:{dependency}"
            f.writelines(f"#SBATCH --dependency={dependency}\n")
        f.writelines(f"#SBATCH -p {partition}\n")
        if account is not None:
            f.writelines(f"#SBATCH -A {account}\n")
        f.writelines(f"#SBATCH --job-name={job_name}\n")
        f.writelines(f"#SBATCH --mem={mem}\n")
        if exclusive:
            f.writelines("#SBATCH --exclusive\n")
        if overcommit:
            f.writelines("#SBATCH --overcommit\n")
        f.writelines(f"#SBATCH --time={time}\n\n")
        f.writelines(f'srun {flags} sh -c "{train_cmd}"\n\n')
        f.writelines("set +x\n")


def run_training(cfg, bignlp_hp_tool_path):
    """
    Main function to launch a training job, with the config given in cfg.
    """
    del cfg["cluster"]["cluster"]
    del cfg["cluster"]["env"]

    hydra_args = convert_to_cli(cfg)

    # Read config
    container_mounts = cfg.container_mounts
    container = cfg.container
    train_cfg = cfg.training
    cluster_cfg = cfg.cluster
    data_dir = cfg.data_dir
    base_results_dir = cfg.base_results_dir
    run_cfg = train_cfg.run

    # Run parameters
    name = run_cfg.name
    results_dir = run_cfg.results_dir
    time_limit = run_cfg.time_limit
    
    if os.path.isdir(results_dir):
        return None
    else:
        os.makedirs(results_dir, exist_ok=True)

    
    # Shared between BCP and BCM 
    scripts_dir = os.path.join(results_dir, "train_scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    new_script_path = os.path.join(scripts_dir, f"{name}.sh")
    code_path = "/opt/bignlp/bignlp-scripts/bignlp/train_scripts/pretrain_gpt.py"
    train_cmd = f"PYTHONPATH=/opt/bignlp/bignlp-scripts:$PYTHONPATH python3 -u {code_path} {hydra_args}"

    nodes = train_cfg.trainer.num_nodes
    ntasks_per_node = train_cfg.trainer.gpus

    # BCM parameters
    partition = cluster_cfg.partition
    account = cluster_cfg.account
    exclusive = cluster_cfg.exclusive
    gpus_per_task = cluster_cfg.gpus_per_task
    job_name_prefix = cluster_cfg.job_name_prefix
    dependency = run_cfg.dependency
    job_name = job_name_prefix + name

    # Process container-mounts.
    mounts_str = f"{bignlp_hp_tool_path}:{bignlp_hp_tool_path},{data_dir}:{data_dir},{base_results_dir}:{base_results_dir}"
    if container_mounts is not None:
        assert isinstance(container_mounts, omegaconf.listconfig.ListConfig), "container_mounts must be a list."
        for mount in container_mounts:
            if mount is not None and isinstance(mount, str):
                mounts_str += f",{mount}:{mount}"

    flags = (
        f"--container-image {container} "
        f"--container-mounts {mounts_str} "
        f"-o {results_dir}/{name}-%j.log "
        f"-e {results_dir}/{name}-%j.error "
    )

    create_slurm_file(
        new_script_path=new_script_path,
        train_cmd=train_cmd,
        job_name=job_name,
        flags=flags,
        dependency=dependency,
        exclusive=exclusive,
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
    dependency = job_id = job_id.decode("utf-8")
    print(f"Submitted Training script with job id: {dependency}")
    return dependency
