import os
import pathlib


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
        f.writelines(f'srun {flags} sh -c \'{train_cmd}\'\n\n')
        f.writelines("set +x\n")


def run_benchmark(
    run_cfg,
    benchmark_cfg,
    cluster_cfg,
    dependency,
    bignlp_scripts_path,
    triton_model_dir,
    model_name,
    container,
    tensor_parallel_size,
    pipeline_parallel_size,
):

    # Run configuration
    model_type = run_cfg.model_type
    model_train_name = model_name
    time_limit = run_cfg.get("time_limit")

    # Benchmark configuration
    tensor_para_size = tensor_parallel_size
    pipeline_para_size = pipeline_parallel_size
    input_len = benchmark_cfg.input_len
    output_len = benchmark_cfg.output_len
    batch_sizes = benchmark_cfg.batch_sizes
    triton_wait_time = benchmark_cfg.triton_wait_time_s

    batch_sizes_str = ' '.join([str(i) for i in batch_sizes])
    task_name = f"inference_benchmark_{model_train_name}_tp{tensor_para_size}_pp{pipeline_para_size}"

    if model_type == "t5":
        input_len_name = "sequence_length"
        output_len_name = "max_output_len"
    elif model_type == "mt5":
        input_len_name = "sequence_length"
        output_len_name = "max_output_len"
    elif model_type == "gpt3":
        input_len_name = "input_lengths"
        output_len_name = "request_output_len"
    else:
        raise Exception(f"Model type: {model_type} not supported")

    # Cluster configuration
    partition = cluster_cfg.get("partition")
    account = cluster_cfg.get("account")
    exclusive = cluster_cfg.get("exclusive")
    job_name_prefix = cluster_cfg.get("job_name_prefix")

    job_name = job_name_prefix + task_name
    nodes = pipeline_para_size
    ntasks_per_node = 1
    gpus_per_task = None

    # Log and results dir
    benchmark_path = f"{bignlp_scripts_path}/bignlp/inference_scripts/benchmark_sweep.sh"
    results_dir = pathlib.Path(run_cfg.results_dir)
    logs_dir = f"{bignlp_scripts_path}/bignlp/inference_scripts/benchmark_sbatch"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    new_script_path = os.path.join(logs_dir, f"{task_name}.sh")

    # Start Triton Server
    gpus = ','.join([str(i) for i in range(0, tensor_para_size)])

    # Benchmark command
    conditional_if_cmd = " if [ $PMIX_RANK = 0 ] && [ \"$PMIX_HOSTNAME\" = \"$SLURMD_NODENAME\" ]; then"

    triton_cmd = conditional_if_cmd + (f" CUDA_VISIBLE_DEVICES={gpus} \\\n"
        "/opt/tritonserver/bin/tritonserver \\\n" 
        f"--model-repository={triton_model_dir} & \\\n"
    )

    benchmark_cmd = triton_cmd + (f"sleep {triton_wait_time} && \\\n"
        f"bash {benchmark_path} \\\n"
        f"{model_train_name} \\\n"
        f"{results_dir} \\\n"
        f"{input_len} \\\n"
        f"{output_len} \\\n"
        f"{input_len_name} \\\n"
        f"{output_len_name} \\\n"
        f"{tensor_para_size} \\\n"
        f"{pipeline_para_size} \\\n"
        f"{batch_sizes_str}; \\\n"
    )

    conditional_benchmark_cmd = benchmark_cmd + (f"else CUDA_VISIBLE_DEVICES={gpus} \\\n"
        "/opt/tritonserver/bin/tritonserver \\\n"
        f"--model-repository={triton_model_dir}; fi"
    )

    # Set Slurm flags
    mounts_str = f"{bignlp_scripts_path}:{bignlp_scripts_path},{results_dir}:{results_dir}"
    flags = (
        "--mpi=pmix "
        f"--container-image {container} "
        f"--container-mounts {mounts_str} "
        f"-o {results_dir}/{task_name}-%j.log "
        f"-e {results_dir}/{task_name}-%j.error "
    )

    create_slurm_file(
        new_script_path=new_script_path,
        train_cmd=conditional_benchmark_cmd,
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

    return new_script_path
