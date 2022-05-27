import sys
import os
import subprocess
import glob

import hydra
from omegaconf import OmegaConf
from bignlp.bignlp_utils import add_container_mounts


def create_slurm_file(
    new_script_path,
    convert_cmd,
    job_name,
    flags="",
    dependency=None,
    time="04:00:00",
    exclusive=True,
    mem=0,
    overcommit=True,
    nodes=1,
    ntasks_per_node=1,
    gpus_per_task=None,
    gpus_per_node=None,
    partition="batch",
    account=None,
):
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
        if mem is not None:
            f.writelines(f"#SBATCH --mem={mem}\n")
        if exclusive:
            f.writelines("#SBATCH --exclusive\n")
        if overcommit:
            f.writelines("#SBATCH --overcommit\n")
        f.writelines(f"#SBATCH --time={time}\n\n")
        f.writelines(f'srun {flags} sh -c "{convert_cmd}"\n\n')
        f.writelines("set +x\n")


def create_bcp_file(cmd_str, num_nodes, ntasks_per_node, log_file, new_script_path):
    with open(new_script_path, "w") as f:
        f.writelines(
            f'bcprun -n {num_nodes} -p {ntasks_per_node} -c "{cmd_str}" >> {log_file} 2>&1\n\n'
        )
        f.writelines("set +x\n")
    os.chmod(new_script_path, 0o755)


def convert_ckpt(cfg, hydra_args="", dependency=None):
    # Read config
    bignlp_path = cfg.get("bignlp_path")
    container = cfg.get("container")
    container_mounts = cfg.get("container_mounts")
    convert_cfg = cfg.get("conversion")
    data_dir = cfg.get("data_dir")
    base_results_dir = cfg.get("base_results_dir")
    run_cfg = convert_cfg.get("run")
    model_cfg = convert_cfg.get("model")

    # Run parameters
    job_name = run_cfg.get("job_name")
    nodes = run_cfg.get("nodes")
    time_limit = run_cfg.get("time_limit")
    ntasks_per_node = run_cfg.get("ntasks_per_node")
    convert_name = run_cfg.get("convert_name")
    model_train_name = run_cfg.get("model_train_name")
    results_dir = run_cfg.get("results_dir")
    output_path = run_cfg.get("output_path")
    nemo_file_name = run_cfg.get("nemo_file_name")
    gpus_per_node = run_cfg.get("ntasks_per_node")
    nemo_file_path = os.path.join(output_path, nemo_file_name)

    # Model parameters
    model_type = model_cfg.get("model_type")
    checkpoint_folder = model_cfg.get("checkpoint_folder")
    checkpoint_name = model_cfg.get("checkpoint_name")
    hparams_file = model_cfg.get("hparams_file")
    tensor_model_parallel_size = model_cfg.get("tensor_model_parallel_size")
    pipeline_model_parallel_size = model_cfg.get("pipeline_model_parallel_size")
    vocab_file = model_cfg.get("vocab_file")
    merge_file = model_cfg.get("merge_file")
    tokenizer_model = model_cfg.get("tokenizer_model")

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    new_script_path = os.path.join(bignlp_path, f"bignlp/conversion_scripts/{model_train_name}.sh")
    code_path = os.path.join(bignlp_path, "bignlp/conversion_scripts/convert_ckpt.py")
    args = (
        f"--gpus_per_node={gpus_per_node} "
        f"--model_type={model_type} "
        f"--checkpoint_folder={checkpoint_folder} "
        f"--checkpoint_name={checkpoint_name} "
        f"--hparams_file={hparams_file} "
        f"--nemo_file_path={nemo_file_path} "
        f"--tensor_model_parallel_size={tensor_model_parallel_size} "
        f"--pipeline_model_parallel_size={pipeline_model_parallel_size} "
        f"--vocab_file={vocab_file} "
        f"--merge_file={merge_file} "
        f"--tokenizer_model={tokenizer_model} "
    )
    if cfg.get("cluster_type") == "bcp":
        args += "--bcp "
    args = args.replace(" ", " \\\n  ")
    cmd_str = f"python3 -u {code_path} \\\n  {args}"

    # Delete conf override file if exists
    hparams_override_file = os.path.join(output_path, "hparams_override.yaml")
    if os.path.exists(hparams_override_file):
        os.remove(hparams_override_file)

    cluster_cfg = cfg.get("cluster")
    if cfg.get("cluster_type") == "bcm":
        # BCM parameters
        partition = cluster_cfg.get("partition")
        account = cluster_cfg.get("account")
        exclusive = cluster_cfg.get("exclusive")
        job_name_prefix = cluster_cfg.get("job_name_prefix")
        gpus_per_task = cluster_cfg.get("gpus_per_task")
        gpus_per_node = cluster_cfg.get("gpus_per_node")

        # Process container-mounts.
        mounts_str = f"{bignlp_path}:{bignlp_path},{data_dir}:{data_dir},{base_results_dir}:{base_results_dir}"
        mounts_str += add_container_mounts(container_mounts)

        flags = (
            f"--no-container-mount-home "
            f"--container-image {container} "
            f"--container-mounts {mounts_str} "
            f"-o {results_dir}/convert-%j.log "
            f"-e {results_dir}/convert-%j.error "
        )
        create_slurm_file(
            new_script_path=new_script_path,
            convert_cmd=cmd_str,
            job_name=job_name_prefix + job_name,
            flags=flags,
            dependency=dependency,
            exclusive=exclusive,
            mem=None,
            overcommit=None,
            time=time_limit,
            nodes=nodes,
            ntasks_per_node=ntasks_per_node,
            gpus_per_task=gpus_per_task,
            gpus_per_node=gpus_per_node,
            partition=partition,
            account=account,
        )
        job_id = subprocess.check_output([f"sbatch --parsable {new_script_path}"], shell=True)
        dependency = job_id.decode("utf-8")
        print(f"Submitted Conversion script with job id: {dependency}")
        return dependency

    elif cfg.get("cluster_type") == "bcp":
        create_bcp_file(
            new_script_path=new_script_path,
            cmd_str=cmd_str,
            num_nodes=nodes,
            ntasks_per_node=ntasks_per_node,
            log_file=f"{results_dir}/convert_log.txt",
        )

        submit_cmd = f"NGC_TASKS_PER_NODE={ntasks_per_node} {new_script_path}"
        job_id = subprocess.check_output([f"{submit_cmd}"], shell=True)
        print(f"Conversion job submitted with command: \n{submit_cmd}")
        return job_id
