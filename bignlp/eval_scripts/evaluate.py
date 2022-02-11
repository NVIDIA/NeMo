import sys
import os
import subprocess

import hydra
import omegaconf


def create_slurm_file(
    new_script_path,
    eval_cmd1,
    eval_cmd2,
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
        if mem is not None:
            f.writelines(f"#SBATCH --mem={mem}\n")
        if exclusive:
            f.writelines("#SBATCH --exclusive\n")
        if overcommit:
            f.writelines("#SBATCH --overcommit\n")
        f.writelines(f"#SBATCH --time={time}\n\n")
        f.writelines(f'srun {flags} --ntasks=1 sh -c "{eval_cmd1}"\n\n')
        f.writelines(f'srun {flags} --ntasks={ntasks_per_node} sh -c "{eval_cmd2}"\n\n')
        f.writelines("set +x\n")


def run_evaluation(cfg, dependency=None):
    # Read config
    bignlp_path = cfg.bignlp_path
    container = cfg.container
    container_mounts = cfg.container_mounts
    eval_cfg = cfg.evaluation
    data_dir = cfg.data_dir
    base_results_dir = cfg.base_results_dir
    run_cfg = eval_cfg.run
    model_cfg = eval_cfg.model

    # Model parameters
    model_type = model_cfg.type
    checkpoint = model_cfg.checkpoint_path
    tensor_model_parallel_size = model_cfg.tensor_model_parallel_size
    batch_size = model_cfg.eval_batch_size
    vocab_file = model_cfg.vocab_file
    merge_file = model_cfg.merge_file

    # Run parameters
    name = run_cfg.name
    time_limit = run_cfg.time_limit
    nodes = run_cfg.nodes
    ntasks_per_node = run_cfg.ntasks_per_node
    gpus_per_task = run_cfg.gpus_per_task
    eval_name = run_cfg.eval_name
    convert_name = run_cfg.convert_name
    model_train_name = run_cfg.model_train_name
    tasks = run_cfg.tasks
    results_dir = run_cfg.results_dir
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Command to download the eval datasets.
    cache_dir = os.path.join(results_dir, "data_cache")
    code_path1 = os.path.join(bignlp_path, "bignlp/eval_scripts/eval_harness/download.py")
    eval_cmd1 = f"python {code_path1} --tasks {tasks} --cache_dir {cache_dir} " \
    
    # Command to run the model on the eval datasets.
    new_script_path = os.path.join(bignlp_path, f"bignlp/eval_scripts/{name}.sh")
    code_path2 = os.path.join(bignlp_path, "bignlp/eval_scripts/eval_harness/evaluate.py")
    eval_cmd2 = f"python -u {code_path2} " \
                f"--name {name} " \
                f"--model {model_type} " \
                f"--tasks {tasks} " \
                f"--cache_dir {cache_dir} " \
                f"--batch_size {batch_size} " \
                f"--output_path {results_dir} " \
                f"--model_args nemo_model={checkpoint},tensor_model_parallel_size={tensor_model_parallel_size},vocab_file={vocab_file},merges_file={merge_file} "

    if cfg.cluster_type == "bcm":
        # BCM parameters
        partition = cfg.cluster.partition
        account = cfg.cluster.account
        exclusive = cfg.cluster.exclusive
        job_name_prefix = cfg.cluster.job_name_prefix
        job_name = os.path.join(job_name_prefix, name)

        # Process container-mounts.
        mounts_str = f"{bignlp_path}:{bignlp_path},{data_dir}:{data_dir},{base_results_dir}:{base_results_dir}"
        if container_mounts is not None:
            assert isinstance(container_mounts, omegaconf.listconfig.ListConfig), "container_mounts must be a list."
            for mount in container_mounts:
                if mount is not None and isinstance(mount, str):
                    mounts_str += f",{mount}:{mount}"

        flags = (
            f"--no-container-mount-home "
            f"--container-image {container} "
            f"--container-mounts {mounts_str} "
            f"-o {results_dir}/{name}-%j.log "
            f"-e {results_dir}/{name}-%j.error "
        )

        create_slurm_file(
            new_script_path=new_script_path,
            eval_cmd1=eval_cmd1,
            eval_cmd2=eval_cmd2,
            job_name=job_name,
            flags=flags,
            dependency=dependency,
            exclusive=exclusive,
            mem=None,
            overcommit=None,
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
        print(f"Submitted Evaluation script with job id: {dependency}")
        return dependency

    elif cfg.cluster_type == "bcp":
        print(f"Evaluation dataset download job submitted with command: \n{eval_cmd1}")
        subprocess.check_output([f"{eval_cmd1}"], shell=True)

        print(f"Evaluation job submitted with command: \n{eval_cmd2}")
        subprocess.check_output([f"{eval_cmd2}"], shell=True)
        return None
