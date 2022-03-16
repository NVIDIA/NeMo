import sys
import os
import subprocess

import hydra
import omegaconf
from bignlp.bignlp_utils import convert_to_cli, add_container_mounts
from bignlp.finetune_scripts.data import download_glue


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


def create_bcp_file(
        train_cmd,
        num_nodes,
        log_file,
        new_script_path
):
    with open(new_script_path, "w") as f:
        f.writelines(f'bcprun -n {num_nodes} -c \"{train_cmd}\" >> {log_file} 2>&1 \n')
        f.writelines("\n")
        f.writelines("set +x \n")
    os.chmod(new_script_path, 0o755)


def run_finetuning(cfg, hydra_args="", dependency=None):
    """
    Main function to launch a training job, with the config given in cfg.
    """
    # Read config
    bignlp_path = cfg.bignlp_path
    container_mounts = cfg.container_mounts
    container = cfg.container
    finetune_cfg = cfg.finetuning
    cluster_cfg = cfg.cluster
    data_dir = cfg.data_dir
    base_results_dir = cfg.base_results_dir
    run_cfg = finetune_cfg.run

    # Run parameters
    name = run_cfg.name
    task_name = run_cfg.task_name
    results_dir = run_cfg.results_dir
    time_limit = run_cfg.time_limit

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    download_glue.download_glue(
        data_dir=os.path.join(data_dir, "glue_data"),
        tasks=task_name
    )

    # Shared between BCP and BCM 
    new_script_path = os.path.join(bignlp_path, f"bignlp/finetune_scripts/{name}.sh")
    code_path = os.path.join(bignlp_path, "bignlp/finetune_scripts/finetune_t5.py")

    hydra_args = hydra_args.replace(" ", " \\\n  ")
    train_cmd = f"PYTHONPATH={bignlp_path}" + ":${PYTHONPATH} \\\n" + f"python3 -u {code_path} \\\n  {hydra_args}"

    nodes = finetune_cfg.trainer.num_nodes
    ntasks_per_node = finetune_cfg.trainer.gpus

    # BCM parameters
    if cfg.cluster_type == "bcm":
        partition = cluster_cfg.partition
        account = cluster_cfg.account
        exclusive = cluster_cfg.exclusive
        gpus_per_task = cluster_cfg.gpus_per_task
        job_name_prefix = cluster_cfg.job_name_prefix
        if dependency is None:
            dependency = run_cfg.dependency
        job_name = job_name_prefix + name

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
            partition=partition,
            account=account,
        )
        job_id = subprocess.check_output(
            [f"sbatch --parsable {new_script_path}"], shell=True
        )
        dependency = job_id = job_id.decode("utf-8")
        print(f"Submitted Finetuning script with job id: {dependency}")
        return dependency

    # BCP parameters
    if cfg.cluster_type == "bcp":
        create_bcp_file(
            new_script_path=new_script_path,
            train_cmd=train_cmd,
            num_nodes=nodes,
            log_file=f"{results_dir}/{name}.log",
        )
        submit_cmd = f"NGC_NTASKS_PER_NODE={ntasks_per_node} {new_script_path}"
        subprocess.check_output([f"{submit_cmd}"], shell=True)
        print(f"Training job submitted with command: \n{submit_cmd}")
        return None
