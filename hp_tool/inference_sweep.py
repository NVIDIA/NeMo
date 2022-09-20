import os
import shutil
import subprocess
import sys

import omegaconf
import pathlib

from hp_tool import utils


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
    model_config_path = (
        f"{bignlp_hp_tool_path}/conf/ft_model_config/{model_type}/{model_train_name}.ini"
    )
    results_dir = run_cfg.get("results_dir")
    results_dir = os.path.join(results_dir, "inference")
    os.makedirs(results_dir, exist_ok=True)
    workspace_dir = os.path.join(results_dir, "workspace")
    os.makedirs(workspace_dir, exist_ok=True)

    # Benchmark configuration
    benchmark_cfg = inference_cfg.get("benchmark")
    max_batch_size = max(benchmark_cfg.batch_sizes)

    # Process container-mounts.
    mounts_str = (
        f"{bignlp_hp_tool_path}:{bignlp_hp_tool_path},{base_results_dir}:{base_results_dir}"
    )
    mounts_str += utils.add_container_mounts(container_mounts)

    container = cfg.get("training_container")
    output_log = os.path.join(workspace_dir, "log-test.out")
    error_log = os.path.join(workspace_dir, "log-test.err")
    flags = (
        f"--container-image {container} "
        f"--container-mounts {mounts_str} "
        f"--no-container-mount-home "
        f"--output {output_log} "
        f"--error {error_log} "
    )

    run = 0
    for tensor_parallel_size in tensor_parallel_sizes:
        for pipeline_parallel_size in pipeline_parallel_sizes:

            benchmark_model_name = (
                f"{model_train_name}_tp{tensor_parallel_size}_pp{pipeline_parallel_size}"
            )
            task_name = f"inference_sweep_{benchmark_model_name}"

            # Prepare trition configuration
            triton_model_dir = (
                f"{results_dir}/model_repo_{tensor_parallel_size}_{pipeline_parallel_size}"
            )
            model_dir = f"{triton_model_dir}/{benchmark_model_name}/1/{tensor_parallel_size}-gpu"
            os.makedirs(model_dir, exist_ok=True)

            shutil.copyfile(model_config_path, f"{model_dir}/config.ini")

            prepare_model_config_script_path = f"{bignlp_scripts_path}/bignlp/collections/export_scripts/prepare_triton_model_config.py"
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
            # subprocess.call(f"{triton_prepare_model_config_cmd}", shell=True)

            # Run benchmark
            benchmark_script = run_benchmark(
                cfg=cfg,
                run_cfg=run_cfg,
                benchmark_cfg=benchmark_cfg,
                cluster_cfg=cluster_cfg,
                dependency=None,
                bignlp_scripts_path=bignlp_scripts_path,
                prepare_cmd=triton_prepare_model_config_cmd,
                triton_model_dir=triton_model_dir,
                model_name=benchmark_model_name,
                flags=flags,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                nodes=1,  # TODO: update to correct number
                workspace_path=workspace_dir,
                verbose=True,
            )

            job_id = subprocess.check_output([f"sbatch --parsable {benchmark_script}"], shell=True)
            job_id = job_id.decode("utf-8")
            print(f"Submitted Training script with job id: {job_id}")
            run += 1


def run_benchmark(
    cfg,
    run_cfg,
    benchmark_cfg,
    cluster_cfg,
    dependency,
    bignlp_scripts_path,
    prepare_cmd,
    triton_model_dir,
    model_name,
    flags,
    tensor_parallel_size,
    pipeline_parallel_size,
    nodes,
    workspace_path,
    verbose=True,
):

    # dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # parser = argparse.ArgumentParser(description="Test BigNLP models")
    # parser.add_argument("--cluster-config-path", help="Path to cluster configuration file", required=True)
    # parser.add_argument("--model-config-path", help="Path to model configuration file", required=True)
    # parser.add_argument("--start-id-path", help="Path to start id csv file", required=True)
    # parser.add_argument(
    #    "--workspace-path",
    #    help="Path to workspace dir where logs and artifacts will be stored",
    #    default=f"./infer_workspace-{dt}",
    # )
    # parser.add_argument("--verbose", "-v", help="Provides verbose output", action="store_true", default=False)
    # args = parser.parse_args()

    workspace_path_absolute = pathlib.Path(workspace_path).resolve().absolute()
    # config_logger(workspace_path_absolute, verbose)

    # LOGGER.info(f"Arguments:")
    # for name, value in vars(args).items():
    #    LOGGER.info(f"    {name}: {value}")

    """
    config_path = pathlib.Path(args.cluster_config_path).resolve().absolute()
    with config_path.open("r") as config_file:
        cluster_config = yaml.load(config_file, Loader=yaml.SafeLoader)
    cluster_dir_path = workspace_path_absolute / CLUSTER_DIR_NAME
    """

    # job_name_prefix = cluster_config["env"]["job_name_prefix"]
    # training_container_image = cluster_cfg["env"]["training_container_image"]

    # variant = Variant.from_triton_model_repository(triton_model_repository_path)
    # LOGGER.info(f"Config variant {variant}")

    # executor = ClusterExecutor(cluster_dir_path=cluster_dir_path, cluster_config=cluster_config["cluster"])

    start_id_path = "start_id.csv"
    output_path = "sweep.out"
    cluster_config_path = os.path.join(cfg.bignlp_hp_tool_path, "conf/cluster/bcm.yaml")

    cmd = (
        f"export CUDA_VISIBLE_DEVICES={','.join(map(str, range(0, 8)))} && "
        f"/opt/bignlp/FasterTransformer/build/bin/multi_gpu_gpt_sweep "
        f"{cluster_config_path} "
        f"{start_id_path} "
        f"{workspace_path}/{output_path} "
        f"{tensor_parallel_size} "
        f"{pipeline_parallel_size} "
    )

    submission_script_path = f"{workspace_path}/submission_script.sh"

    job_name = "ft_benchmark"

    utils.create_slurm_file(
        new_script_path=submission_script_path,
        cmds=[prepare_cmd, cmd],
        job_name=f"{cluster_cfg.job_name_prefix}{job_name}",
        flags=flags,
        dependency=None,  # TODO: add dependency when splitting commands into two.
        exclusive=cluster_cfg.exclusive,
        mem=None,
        overcommit=False,
        # time=time_limit,
        nodes=nodes,
        ntasks=cluster_cfg.get("ntasks"),
        ntasks_per_node=cluster_cfg.get("ntasks_per_node"),
        gpus_per_task=cluster_cfg.get("gpus_per_task"),
        gpus_per_node=cluster_cfg.get("gpus_per_node"),
        partition=cluster_cfg.partition,
        account=cluster_cfg.account,
        exclude=cluster_cfg.get("exclude"),
    )
    return submission_script_path


# f"""
##!/usr/bin/env bash

## parameters
##SBATCH --account=joc
##SBATCH --comment='Triton Inference Server'
##SBATCH --job-name=joc-bermuda:tritonserver_set_ft_175B_random_io_200_16_vocab_128k-type_GPT-tp_8-pp_1-data_fp16-int8_0-maxseq_216-mbs_1
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --open-mode=append
##SBATCH --output={workspace_path}/cluster_workspace/submission_%j.out
##SBATCH --partition=interactive
##SBATCH --signal=USR1@90
##SBATCH --time=0-02:00:00

## command
# srun --mpi pmix \
#  --output /lustre/fsw/joc/piotrm/src/megatron/my_helper_scripts/TEST_WORKSPACE_RANDOM_20220623_175b_vocab128k_pp_1/cluster_workspace/submission_%j.out \
#  --container-image gitlab-master.nvidia.com#dl/dgx/bignlp/infer:main-py3-base \
#  --container-mounts /lustre/fsw/joc/piotrm/src/megatron/my_helper_scripts/TEST_WORKSPACE_RANDOM_20220623_175b_vocab128k_pp_1:/lustre/fsw/joc/piotrm/src/megatron/my_helper_scripts/TEST_WORKSPACE_RANDOM_20220623_175b_vocab128k_pp_1 \
#  --unbuffered \
#  bash -c 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export NCCL_LAUNCH_MODE=GROUP && echo ${SLURM_PROCID}.${SLURM_LOCALID}@$(hostname) && tritonserver --model-repository /lustre/fsw/joc/piotrm/src/megatron/my_helper_scripts/TEST_WORKSPACE_RANDOM_20220623_175b_vocab128k_pp_1/model_repo_ft_175B_random_io_200_16_vocab_128k-type_GPT-tp_8-pp_1-data_fp16-int8_0-maxseq_216-mbs_1
# """

#    benchmark_set_job_def = JobDefinition(
#        name=f"{job_name_prefix}gpt benchmark",
#        description=f"Run gpt benchmark",
#        max_time_s=DEFAULT_BENCHMARK_TIME_MIN * MIN2S,
#        container_image=training_container_image,
#        commands=[
#            f"export CUDA_VISIBLE_DEVICES={','.join(map(str, range(0, 8)))}",
#            f"/opt/bignlp/FasterTransformer/build/bin/multi_gpu_gpt_sweep ",
#            f"{config_path} ",
#            f"{start_id_path} ",
#            f"{workspace_path}/{output_path} ",
#            f"{tensor_parallel_size} ",
#            f"{pipeline_parallel_size}",
#        ],
##         directories_to_mount=[triton_model_repository_path],
#        ports=[DEFAULT_HTTP_PORT, DEFAULT_GRPC_PORT, DEFAULT_METRIC_PORT],
#        tasks_number=pipeline_parallel_size,
#        tasks_number_per_node=1,
#        gpus_number_per_task=tensor_parallel_size,
#    )
#    LOGGER.info(f"[-] Submitted job for {benchmark_set_job_def.description}")
# benchmark_set_job = executor.submit(benchmark_set_job_def)
# benchmark_set_job.wait()
