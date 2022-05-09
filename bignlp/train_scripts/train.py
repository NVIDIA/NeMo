import sys
import os
import subprocess

import hydra
import omegaconf
from bignlp.bignlp_utils import convert_to_cli, add_container_mounts


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
    gpus_per_task=None,
    gpus_per_node=None,
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
        if gpus_per_node is not None:
            f.writelines(f"#SBATCH --gpus-per-node={gpus_per_node}\n")
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


def create_bcp_file(
    train_cmd,
    num_nodes,
    log_file,
    new_script_path,
    env_exports=None,
):
    with open(new_script_path, "w") as f:
        if env_exports is not None:
            env_cmd = f"--env {env_exports}"
        f.writelines(f'bcprun -n {num_nodes} {env_cmd} -c \"{train_cmd}\" >> {log_file} 2>&1 \n')
        f.writelines("\n")
        f.writelines("set +x \n")
    os.chmod(new_script_path, 0o755)


def run_training(cfg, hydra_args="", dependency=None):
    """
    Main function to launch a training job, with the config given in cfg.
    """
    # Read config
    bignlp_path = cfg.get("bignlp_path")
    container_mounts = cfg.get("container_mounts")
    container = cfg.get("container")
    train_cfg = cfg.get("training")
    cluster_cfg = cfg.get("cluster")
    data_dir = cfg.get("data_dir")
    base_results_dir = cfg.get("base_results_dir")
    run_cfg = train_cfg.get("run")

    # Run parameters
    name = run_cfg.get("name")
    results_dir = run_cfg.get("results_dir")
    time_limit = run_cfg.get("time_limit")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Shared between BCP and BCM
    new_script_path = os.path.join(bignlp_path, f"bignlp/train_scripts/{name}.sh")
    training_config = cfg.get("training_config")
    if "gpt" in training_config:
        code_path = os.path.join(bignlp_path, "bignlp/train_scripts/pretrain_gpt.py")
    elif "t5" in training_config:
        code_path = os.path.join(bignlp_path, "bignlp/train_scripts/pretrain_t5.py")
    else:
        raise ValueError(f"Unrecognized model type in training config `{cfg.training_config}`.")

    hydra_args = hydra_args.replace(" ", " \\\n  ")
    base_cmd = f"python3 -u {code_path} \\\n  {hydra_args}"

    nodes = train_cfg.get("trainer").get("num_nodes")
    ntasks_per_node = train_cfg.get("trainer").get("devices")

    # BCM parameters
    if cfg.cluster_type == "bcm":
        partition = cluster_cfg.get("partition")
        account = cluster_cfg.get("account")
        exclusive = cluster_cfg.get("exclusive")
        gpus_per_task = cluster_cfg.get("gpus_per_task")
        gpus_per_node = cluster_cfg.get("gpus_per_node")
        job_name_prefix = cluster_cfg.get("job_name_prefix")

        if dependency is None:
            dependency = run_cfg.get("dependency")
        job_name = job_name_prefix + name

        train_cmd = f"PYTHONPATH={bignlp_path}:${{PYTHONPATH}} \\\n {base_cmd}"

        # Process container-mounts.
        mounts_str = f"{bignlp_path}:{bignlp_path},{data_dir}:{data_dir},{base_results_dir}:{base_results_dir}"
        mounts_str += add_container_mounts(container_mounts)

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
            gpus_per_node=gpus_per_node,
            partition=partition,
            account=account,
        )
        job_id = subprocess.check_output([f"sbatch --parsable {new_script_path}"], shell=True)
        dependency = job_id = job_id.decode("utf-8")
        print(f"Submitted Training script with job id: {dependency}")
        return dependency

    # BCP parameters
    if cfg.get("cluster_type") == "bcp":
        env_exports = f"PYTHONPATH=${{PYTHONPATH}}:{bignlp_path}"
        create_bcp_file(
            new_script_path=new_script_path,
            train_cmd=base_cmd,
            num_nodes=nodes,
            log_file=f"{results_dir}/{name}.log",
            env_exports=env_exports,
        )
        submit_cmd = f"NGC_NTASKS_PER_NODE={ntasks_per_node} {new_script_path}"
        subprocess.check_output([f"{submit_cmd}"], shell=True)
        print(f"Training job submitted with command: \n{submit_cmd}")
        return None
