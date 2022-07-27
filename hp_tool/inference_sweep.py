import os
import shutil
import subprocess
import sys

import omegaconf


def search_inference_config(base_cfg, cfg):
    """
    Main function to launch a inference sweep job, with the config given in cfg.
    """
    # Read config
    bignlp_hp_tool_path = cfg.get("bignlp_hp_tool_path")
    bignlp_scripts_path = cfg.get("bignlp_scripts_path")
    container_mounts = cfg.get("container_mounts")
    container = cfg.get("inference_container")
    hp_cfg = cfg.get("search_config")
    base_results_dir = cfg.get("base_results_dir")

    # Cluster parameters
    cluster_cfg = cfg.get("cluster")

    # Inference settings
    inference_cfg = hp_cfg.get("inference_settings")

    # Run configuration
    run_cfg = inference_cfg.get("run")
    model_type = run_cfg.model_type
    model_train_name = run_cfg.get("model_train_name")
    tensor_parallel_sizes = run_cfg.tensor_parallel_sizes
    pipeline_parallel_sizes = run_cfg.pipeline_parallel_sizes
    triton_dir = f"{run_cfg.results_dir}/model_repo"
    model_config_path = f"{bignlp_hp_tool_path}/conf/ft_model_config/{model_type}/{model_train_name}.ini"
    results_dir = run_cfg.get("results_dir")
    os.makedirs(results_dir, exist_ok=True)

    # Benchmark configuration
    benchmark_cfg = inference_cfg.get("benchmark")
    max_batch_size = max(benchmark_cfg.batch_sizes)

    sys.path.append(bignlp_scripts_path)
    from bignlp.inference_scripts.benchmark import run_benchmark

    # Process container-mounts.
    mounts_str = f"{bignlp_hp_tool_path}:{bignlp_hp_tool_path},{base_results_dir}:{base_results_dir}"
    if container_mounts is not None:
        assert isinstance(
            container_mounts, omegaconf.listconfig.ListConfig
        ), "container_mounts must be a list."
        for mount in container_mounts:
            if mount is not None and isinstance(mount, str):
                mounts_str += f",{mount}:{mount}"

    run = 0
    for tensor_parallel_size in tensor_parallel_sizes:
        for pipeline_parallel_size in pipeline_parallel_sizes:

            benchmark_model_name = f"{model_train_name}_tp{tensor_parallel_size}_pp{pipeline_parallel_size}"
            task_name = f"inference_sweep_{benchmark_model_name}"

            # Prepare trition configuration
            triton_model_dir = f"{results_dir}/model_repo_{run}"
            model_dir = f"{triton_model_dir}/{benchmark_model_name}/1/{tensor_parallel_size}-gpu"
            os.makedirs(model_dir, exist_ok=True)

            shutil.copyfile(model_config_path, f"{model_dir}/config.ini")

            prepare_model_config_script_path = f"{bignlp_scripts_path}/bignlp/export_scripts/prepare_triton_model_config.py"
            template_path = f"{bignlp_hp_tool_path}/conf/triton_config/{model_type}/config.pbtxt"

            triton_prepare_model_config_cmd = (
                f"python3 -u {prepare_model_config_script_path}"
                f" --model-train-name {benchmark_model_name}"
                f" --template-path {template_path}"
                f" --ft-checkpoint {model_dir}"
                f" --config-path {triton_model_dir}/{benchmark_model_name}/config.pbtxt"
                f" --max-batch-size {max_batch_size}"
                f" --pipeline-model-parallel-size {pipeline_parallel_size}"
                f" --tensor-model-parallel-size {tensor_parallel_size}"
                f" --data-type {run_cfg.data_type}"
            )
            subprocess.call(f"{triton_prepare_model_config_cmd}", shell=True)

            # Run benchmark
            benchmark_script = run_benchmark(
                run_cfg=run_cfg,
                benchmark_cfg=benchmark_cfg,
                cluster_cfg=cluster_cfg,
                dependency=None,
                bignlp_scripts_path=bignlp_scripts_path,
                triton_model_dir=triton_model_dir,
                model_name=benchmark_model_name,
                container=container,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size
            )

            job_id = subprocess.check_output([f"sbatch --parsable {benchmark_script}"], shell=True)
            job_id = job_id.decode("utf-8")
            print(f"Submitted Training script with job id: {job_id}")
            run += 1
