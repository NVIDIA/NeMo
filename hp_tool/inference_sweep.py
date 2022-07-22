import os
import sys

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

    # Cluster paramters
    cluster_cfg = cfg.get("cluster")

    # Inference settings
    inference_cfg = hp_cfg.get("inference_settings")
    run_cfg = inference_cfg.get("run")
    model_train_name = run_cfg.get("model_size")
    triton_dir = f"{run_cfg.results_dir}/model_repo"
    model_config_path = f"{bignlp_hp_tool_path}/model_config/{run_cfg.model_type}/{model_train_name}.ini"
    benchmark_cfg = inference_cfg.get("benchmark")

    results_dir = os.path.join(run_cfg.get("results_dir"), "sweep")
    os.makedirs(results_dir, exist_ok=True)

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

    for tensor_parallel_size in tensor_parallel_sizes:
        for pipeline_parallel_size in pipeline_parallel_sizes:

            task_name = "inference_sweep_" + model_name + f"_tp{tensor_parallel_size}_pp{pipeline_parallel_size}"
            benchmark_model_name = f"{model_train_name}_tp{tensor_parallel_size}_pp{pipeline_parallel_size}"

            # Prepare trition configuration
            model_dir = f"{results_dir}/model-repo/{benchmark_model_name}/1/{tensor_parallel_size}-gpu"
            os.makedirs(model_dir, exist_ok=True)

            shutil.copyfile(model_config_path, f"{model_dir}/config.ini")

            prepare_model_config_script_path = f"{bignlp-scripts-path}/bignlp/export/prepare_mode_config.py"
            template_path = f"{bignlp_hp_tool_path}/conf/triton_config/{model_name}/config.pbtxt"

            triton_prepare_model_config_cmd = (
                f"python -u {prepare_model_config_script_path} \\\n"
                f" --model-train-name {model-train-name} \\\n"
                f" --template-path {template_path} \\\n"
                f" --ft-checkpoint {model_dir} \\\n"
                f" --config-path {triton_model_dir}/config.pbtxt \\\n"
                f" --max-batch-size {triton_cfg.max_batch_size} \\\n"
                f" --pipeline-model-parallel-size {pipeline_parallel_size} \\\n"
                f" --data-type {triton_cfg.data_type}"
            )
            subprocess.run([f"{triton_prepare_model_config_cmd}"])

            benchmark_cfg["container"] = container
            benchmark_cfg["tensor_model_parallel_size"] = tensor_parallel_size
            benchmark_cfg["pipeline_model_parallel_size"] = pipeline_parallel_size

            # Run benchmark
            benchmark_script = run_benchmark(
                cfg=cfg,
                run_cfg=run_cfg,
                benchmark_cfg=benchmark_cfg,
                cluster_cfg=cluster_cfg,
                dependency=None,
                model_path=f"{model_dir}"
            )

            job_id = subprocess.check_output([f"sbatch --parsable {benchmark_script}"], shell=True)
            job_id = job_id.decode("utf-8")
            print(f"Submitted Training script with job id: {job_id}")

    return dependency

