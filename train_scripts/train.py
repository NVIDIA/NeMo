import sys
import os
import subprocess

import hydra
import omegaconf


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


def create_bcp_submit_cmd(
    job_name,
    container,
    workspace_common,
    workspace_scripts,
    bignlp_path,
    bcp_script,
    instance,
    num_nodes,
    ntasks_per_node=8,
    array_type="pytorch",
    total_runtime="10h"
):
    base_cmd = f"cd {bignlp_path}; NGC_NTASKS_PER_NODE={ntasks_per_node} {bcp_script}"
    if (num_nodes == 1):
                num_nodes = 2  # bcprun needs at least 2 nodes    
    submit_cmd = f"ngc batch run --name \"{job_name}\" --image \"{container}\" \
    --commandline \"{base_cmd}\" --workspace {workspace_common}:/workspace-common \
    --workspace {workspace_scripts}:/workspace-scripts --result /result \
    --preempt RUNONCE --instance {instance} --replicas {num_nodes} \
    --array-type {array_type} --total-runtime {total_runtime}"
    
    return submit_cmd

def create_bcp_file(
    bignlp_path,
    train_cmd,
    num_nodes,
    log_file,
    err_file,
    new_script_path
):
    with open(new_script_path, "w") as f:
        # Replace bcprun by {bignlp_path}/bcprun2 if latest bcprun with local-rank fix is not deployed
        f.writelines(f'bcprun -n {num_nodes} -c \"{train_cmd}\" >> {log_file} 2>>{err_file} \n')
        f.writelines("\n")
        f.writelines("set +x \n") 
    os.chmod(new_script_path, 0o755)

def run_training(cfg, hydra_args="", dependency=None):
    """
    Main function to launch a training job, with the config given in cfg.
    """
    # Read config
    bignlp_path = cfg.bignlp_path
    container_mounts = cfg.container_mounts
    container = cfg.container
    train_cfg = cfg.training
    cluster_cfg = cfg.cluster
    run_cfg = train_cfg.run

    # Run parameters
    name = run_cfg.name
    results_dir = run_cfg.results_dir
    log_dir = run_cfg.log_dir
    time_limit = run_cfg.time_limit
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Shared between BCP and BCM 
    new_script_path = os.path.join(bignlp_path, f"train_scripts/{name}.sh")
    code_path = os.path.join(bignlp_path, "train_scripts/pretrain_gpt.py")
    train_cmd = f"python3 -u {code_path} {hydra_args}"

    nodes = train_cfg.trainer.num_nodes
    ntasks_per_node = train_cfg.trainer.gpus
    gpus_per_task = ntasks_per_node

    # BCM parameters
    if cfg.cluster_type == "bcm":
        slurm_cfg = cluster_cfg.slurm
        partition = slurm_cfg.partition
        account = slurm_cfg.account
        exclusive = slurm_cfg.exclusive
        job_name_prefix = slurm_cfg.job_name_prefix
        if dependency is None:
            dependency = run_cfg.dependency
        job_name = job_name_prefix + name

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
            f"-o {log_dir}/{name}-%j.log "
            f"-e {log_dir}/{name}-%j.error "
        )

        create_slurm_file(
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
        dependency = job_id = job_id.decode("utf-8")
        print(f"Submitted Training script with job id: {dependency}")
        return dependency

    # BCP parameters
    if cfg.cluster_type == "bcp":
        bcp_cfg = cluster_cfg.bcp
        instance = bcp_cfg.instance
        job_name_prefix = bcp_cfg.job_name_prefix
        job_name = job_name_prefix + name

        create_bcp_file(
            bignlp_path=bignlp_path,
            new_script_path=new_script_path,
            train_cmd=train_cmd,
            num_nodes=nodes,
            log_file=f"{log_dir}/log.txt",
            err_file=f"{log_dir}/err.txt"
        )

        submit_cmd = create_bcp_submit_cmd(
            job_name=job_name,
            container=container,
            workspace_common=bcp_cfg.workspace_common,
            workspace_scripts=bcp_cfg.workspace_scripts,
            bignlp_path=bignlp_path,
            bcp_script=new_script_path,
            instance=instance,
            num_nodes=nodes,
            ntasks_per_node=ntasks_per_node,
            array_type="PYTORCH",
            total_runtime=run_cfg.time_limit_bcp
        )
        print(f"\n Submit command after data is ready:\n {submit_cmd}")
        print(f"\n Script file: {new_script_path}")
