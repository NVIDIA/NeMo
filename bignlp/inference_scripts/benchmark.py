import os
import shutil
import subprocess


def append_triton_parameters(config, tensor_para_size, pipeline_para_size, data_type, all_reduce, model_type, ckpt_path):

    parameter_str = f"""
        parameters {{
          key: "tensor_para_size"
          value: {{
            string_value: \"{tensor_para_size}\"
          }}
        }}
        parameters {{
          key: "pipeline_para_size"
          value: {{
            string_value: "{pipeline_para_size}"
          }}
        }}
        parameters {{
          key: "data_type"
          value: {{
            string_value: "{data_type}"
          }}
        }}
        parameters {{
          key: "enable_custom_all_reduce"
          value: {{
            string_value: "{all_reduce}"
          }}
        }}
        parameters {{
          key: "model_type"
          value: {{
            string_value: "{model_type}"
          }}
        }}
        parameters {{
          key: "model_checkpoint_path"
          value: {{
            string_value: "{ckpt_path}"
          }}
        }}"""

    with open(config, "a") as config_file:
        config_file.write(parameter_str)


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


def run_benchmark():

    # Configuration
    model_type = "t5"
    model_size = "23b"
    tensor_para_size = 4
    pipeline_para_size = 1
    input_len = 60
    output_len = 20
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    batch_sizes_str = ' '.join([str(i) for i in batch_sizes])
    model_path = None
    bignlp_scripts_path = "/lustre/fsw/joc/donghyukc/bignlp-scripts"
    container = "gitlab-master.nvidia.com#dl/dgx/bignlp/infer:infer_update-py3-base"
    triton_wait_time = 30

    task_name = f"inference_benchmark_{model_type}_{model_size}_tp{tensor_para_size}_pp{pipeline_para_size}"

    # Cluster configuration
    partition = "luna"
    account = "joc"
    time_limit = "0:30:00"
    exclusive = True
    job_name_prefix = "joc-bignlp_inference:"
    job_name = job_name_prefix + task_name
    nodes = 1
    ntasks_per_node = 1
    gpus_per_task = None
    dependency = None

    # Log and results dir
    benchmark_path = f"{bignlp_scripts_path}/bignlp/inference_scripts/benchmark_sweep.sh"

    results_dir = f"{bignlp_scripts_path}/results/inference/benchmark/{model_type}"
    logs_dir = f"{bignlp_scripts_path}/bignlp/inference_scripts/benchmark"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    triton_path = f"{results_dir}/triton/fastertransformer"
    os.makedirs(triton_path, exist_ok=True)

    new_script_path = os.path.join(logs_dir, f"{task_name}.sh")

    # Check if model path exists
    if model_path is None:
        model_path = f"{triton_path}/1/8-gpu"
        os.makedirs(model_path, exist_ok=True)

        model_config = f"{bignlp_scripts_path}/bignlp/inference_scripts/model_config/{model_type}/config_{model_size}.ini"
        shutil.copyfile(model_config, f"{model_path}/config.ini")

    # Generate triton configuration
    triton_template_config = f"{bignlp_scripts_path}/bignlp/inference_scripts/triton_config/{model_type}/config.pbtxt"
    triton_config = f"{triton_path}/config.pbtxt" 
    shutil.copyfile(triton_template_config, triton_config)
    append_triton_parameters(triton_config, tensor_para_size, pipeline_para_size, "fp16", 0, "T5", model_path)

    # Start Triton Server
    gpus = ','.join([str(i) for i in range(0, tensor_para_size)])
    triton_cmd = (f"CUDA_VISIBLE_DEVICES={gpus} \\\n"
        f"mpirun -n {pipeline_para_size} --allow-run-as-root \\\n" 
        "/opt/tritonserver/bin/tritonserver \\\n" 
        f"--model-repository={results_dir}/triton & \\\n"
    )

    # Run benchmark command
    benchmark_cmd = triton_cmd + (f"sleep {triton_wait_time} && \\\n"
        f"bash {benchmark_path} \\\n"
        f"{results_dir} \\\n"
        f"{input_len} \\\n"
        f"{output_len} \\\n"
        f"{tensor_para_size} \\\n"
        f"{pipeline_para_size} \\\n"
        f"{batch_sizes_str}"
    )

    # Set Slurm flags
    mounts_str = f"{bignlp_scripts_path}:{bignlp_scripts_path},{results_dir}:{results_dir}"
    flags = (
        f"--container-image {container} "
        f"--container-mounts {mounts_str} "
        f"-o {results_dir}/{task_name}-%j.log "
        f"-e {results_dir}/{task_name}-%j.error "
    )

    create_slurm_file(
        new_script_path=new_script_path,
        train_cmd=benchmark_cmd,
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

    job_id = subprocess.check_output([f"sbatch --parsable {new_script_path}"], shell=True)
    job_id = job_id.decode("utf-8")
    print(f"Submitted Training script with job id: {job_id}")


if __name__=="__main__":
    run_benchmark()
