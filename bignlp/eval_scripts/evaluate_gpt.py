import sys
import os
import subprocess

import hydra
import omegaconf
from bignlp.bignlp_utils import add_container_mounts


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
        f.writelines(f'srun {flags} sh -c "{eval_cmd1}"\n\n')
        f.writelines(f'srun {flags} sh -c "{eval_cmd2}"\n\n')
        f.writelines("set +x\n")


def create_bcp_file(
        cmd_str1,
        cmd_str2,
        num_nodes,
        ntasks_per_node,
        log_file,
        new_script_path
):
    with open(new_script_path, "w") as f:
        f.writelines(f'bcprun -n {num_nodes} -p {ntasks_per_node} -c "{cmd_str1}; {cmd_str2}" >> {log_file} 2>&1\n\n')
        f.writelines("set +x\n")
    os.chmod(new_script_path, 0o755)


def run_evaluation(cfg, dependency=None):
    # Read config
    bignlp_path = cfg.get("bignlp_path")
    container = cfg.get("container")
    container_mounts = cfg.get("container_mounts")
    eval_cfg = cfg.get("evaluation")
    data_dir = cfg.get("data_dir")
    base_results_dir = cfg.get("base_results_dir")
    run_cfg = eval_cfg.get("run")
    model_cfg = eval_cfg.get("model")
    cluster_cfg = cfg.get("cluster")

    # Model parameters
    model_type = model_cfg.get("model_type")
    checkpoint_folder = model_cfg.get("checkpoint_folder")
    checkpoint_name = model_cfg.get("checkpoint_name")
    hparams_file = model_cfg.get("hparams_file")
    tensor_model_parallel_size = model_cfg.get("tensor_model_parallel_size")
    pipeline_model_parallel_size = model_cfg.get("pipeline_model_parallel_size")
    precision = model_cfg.get("precision")
    batch_size = model_cfg.get("eval_batch_size")
    vocab_file = model_cfg.get("vocab_file")
    merge_file = model_cfg.get("merge_file")

    # Run parameters
    name = run_cfg.get("name")
    time_limit = run_cfg.get("time_limit")
    nodes = run_cfg.get("nodes")
    ntasks_per_node = run_cfg.get("ntasks_per_node")
    gpus_per_task = cluster_cfg.gpus_per_task
    eval_name = run_cfg.get("eval_name")
    convert_name = run_cfg.get("convert_name")
    model_train_name = run_cfg.get("model_train_name")
    tasks = run_cfg.get("tasks")
    results_dir = run_cfg.get("results_dir")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Command to download the eval datasets.
    cache_dir = os.path.join(results_dir, "data_cache")
    code_path1 = os.path.join(bignlp_path, "bignlp/eval_scripts/eval_harness/download.py")
    eval_cmd1 = f"python {code_path1} \\\n  --tasks {tasks} \\\n  --cache_dir {cache_dir}"

    # Command to run the model on the eval datasets.
    new_script_path = os.path.join(bignlp_path, f"bignlp/eval_scripts/{name}.sh")
    code_path2 = os.path.join(bignlp_path, "bignlp/eval_scripts/eval_harness/evaluate.py")
    args = f"--name={name} " \
           f"--model={model_type} " \
           f"--tasks={tasks} " \
           f"--cache_dir={cache_dir} " \
           f"--batch_size={batch_size} " \
           f"--output_path={results_dir} " \
           f"--vocab_file={vocab_file} " \
           f"--merge_file={merge_file} " \
           f"--checkpoint_folder={checkpoint_folder} " \
           f"--checkpoint_name={checkpoint_name} " \
           f"--hparams_file={hparams_file} " \
           f"--pipeline_model_parallel_size={pipeline_model_parallel_size} " \
           f"--tensor_model_parallel_size={tensor_model_parallel_size} " \
           f"--precision={precision}"
    args = args.replace(" ", " \\\n  ")
    eval_cmd2 = f"python -u {code_path2} \\\n {args}"

    cluster_cfg = cfg.get("cluster")
    if cfg.get("cluster_type") == "bcm":
        # BCM parameters
        partition = cluster_cfg.get("partition")
        account = cluster_cfg.get("account")
        exclusive = cluster_cfg.get("exclusive")
        job_name_prefix = cluster_cfg.get("job_name_prefix")
        job_name = os.path.join(job_name_prefix, name)

        # Process container-mounts.
        mounts_str = f"{bignlp_path}:{bignlp_path},{data_dir}:{data_dir},{base_results_dir}:{base_results_dir}"
        mounts_str += add_container_mounts(container_mounts)

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

    elif cfg.get("cluster_type") == "bcp":
        create_bcp_file(
            new_script_path=new_script_path,
            cmd_str1=eval_cmd1,
            cmd_str2=eval_cmd2,
            num_nodes=nodes,
            ntasks_per_node=ntasks_per_node,
            log_file=f"{results_dir}/eval_log.txt",
        )

        submit_cmd = f"NGC_TASKS_PER_NODE={ntasks_per_node} {new_script_path}"
        job_id = subprocess.check_output([f"{submit_cmd}"], shell=True)
        print(f"Evaluation job submitted with command: \n{submit_cmd}")
        return job_id
