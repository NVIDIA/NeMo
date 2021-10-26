#!/usr/bin/env python3
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import dataclasses
import datetime
import logging
import pathlib
import sys
import typing

import yaml

from inference_lib.inference import (
    DEFAULT_BENCHMARK_TIME_MIN,
    DEFAULT_MAX_CONFIG_TIME_MIN,
    Paths,
    TritonServerSet,
    Variant,
    convert_model,
    get_dirs_to_mount,
    run_perf_test,
    triton_config_model,
)
from inference_lib.slurm import (
    TRITON_MODEL_REPOSITORY,
    ContainerImageType,
    PyxisExecutor,
    PyxisTritonExecutor,
    get_common_slurm_parameters,
    setup_job,
)
from inference_lib.utils import config_logger, monkeypatch_submitit

LOGGER = logging.getLogger("test_model")


def _convert_model(
    *,
    config,
    paths,
    model_name,
    verbose: bool = False,
):
    conversion_job = None

    def _init_executor():
        dirs_to_mount = get_dirs_to_mount(paths=paths, triton_model_repository_readonly=False)
        slurm_common_parameters = get_common_slurm_parameters(
            cluster_config=config,
            dirs_to_mount=dirs_to_mount,
            container_image_type=ContainerImageType.TRAINING,
        )
        executor_ = PyxisExecutor(folder=paths.submit_dir_path)
        executor_.update_parameters(
            **slurm_common_parameters,
            nodes=1,
            time=DEFAULT_MAX_CONFIG_TIME_MIN,
            job_name=f"joc-bermuda:convert_model",
            comment="Task for converting Megatron/NeMo model to Fastertransformer format",
            setup=["export NO_COLOR=1", f"export PYTHONPATH={sys.path[0]}"],
        )
        if config["slurm"].get("enable_gpus_allocation", True):
            executor_.update_parameters(
                gpus_per_node=8,
            )
        return executor_

    try:
        executor = _init_executor()
        parameters_to_override = {}
        model_path = paths.original_model_path
        if not model_path.suffix:
            parameters_to_override["model_format"] = "megatron"

        conversion_job = executor.submit(
            convert_model,
            config_path=paths.navigator_config_path,
            workspace_path=paths.workspace_path,
            model_name=model_name,
            model_path=model_path,
            verbose=verbose,
            parameters_to_override=parameters_to_override,
        )
        setup_job(conversion_job)
        job_name = conversion_job.get_info().get("JobName")
        LOGGER.info(f"[{conversion_job.job_id}/{job_name}] Submitted conversion for model {paths.original_model_path}")
        LOGGER.info(f"[{conversion_job.job_id}/{job_name}] logs: {conversion_job.paths.stdout}")
        converted_model_path = conversion_job.result()
        LOGGER.info(f"[{conversion_job.job_id}/{job_name}] converted model: {converted_model_path}")
        return pathlib.Path(converted_model_path)
    finally:
        if conversion_job is not None:
            conversion_job.cancel()


def _prepare_triton_model_repository(
    *,
    config,
    paths,
    model_name,
    verbose: bool = False,
):
    preparation_job = None

    def _init_executor():
        dirs_to_mount = get_dirs_to_mount(paths=paths, triton_model_repository_readonly=False)
        slurm_common_parameters = get_common_slurm_parameters(
            cluster_config=config,
            dirs_to_mount=dirs_to_mount,
            container_image_type=ContainerImageType.TRAINING,
        )
        executor_ = PyxisExecutor(folder=paths.submit_dir_path)
        executor_.update_parameters(
            **slurm_common_parameters,
            nodes=1,
            time=DEFAULT_MAX_CONFIG_TIME_MIN,
            job_name="joc-bermuda:config_model",
            comment="Task for preparing Triton Inference Server Model Repository "
            "with model in FasterTransformer format",
            setup=["export NO_COLOR=1", f"export PYTHONPATH={sys.path[0]}"],
        )
        if config["slurm"].get("enable_gpus_allocation", True):
            executor_.update_parameters(
                gpus_per_node=8,
            )
        return executor_

    try:
        executor = _init_executor()

        variant = Variant(model_name=model_name)

        preparation_job = executor.submit(
            triton_config_model,
            config_path=paths.navigator_config_path,
            workspace_path=paths.workspace_path,
            model_path=paths.converted_model_path,
            variants=[variant],
            verbose=verbose,
        )
        setup_job(preparation_job)
        job_name = preparation_job.get_info().get("JobName")
        LOGGER.info(
            f"[{preparation_job.job_id}/{job_name}] Submitted task for preparation set of Triton Model Repositories "
            f"for model {paths.converted_model_path}"
        )

        model_repository_paths = preparation_job.result()
        LOGGER.info(f"[{preparation_job.job_id}/{job_name}] Triton Model Repositories:")
        for model_repository_path in model_repository_paths:
            LOGGER.info(f"[{preparation_job.job_id}/{job_name}]     - {model_repository_path}")
        return model_repository_paths[0] if model_repository_paths else None
    finally:
        if preparation_job is not None:
            preparation_job.cancel()


def _load_triton_model_and_run_benchmark(
    *,
    paths: Paths,
    config: typing.Dict,
    dataset_dir: pathlib.Path,
    verbose: bool = False,
):
    triton_server_set = None
    perf_job = None

    triton_config_path = list(paths.host_triton_model_repository_path.rglob("config.pbtxt"))[0]
    variant = Variant.from_triton_config(triton_config_path)

    def _init_triton_set():
        dirs_to_mount = get_dirs_to_mount(
            paths=paths,
            triton_model_repository_readonly=True,
        )
        slurm_common_parameters = get_common_slurm_parameters(
            cluster_config=config,
            dirs_to_mount=dirs_to_mount,
            container_image_type=ContainerImageType.INFERENCE,
        )

        triton_executor = PyxisTritonExecutor(folder=paths.submit_dir_path)
        triton_executor.update_parameters(
            **slurm_common_parameters,
            time=DEFAULT_BENCHMARK_TIME_MIN,
            comment="Task for handling set of Triton Inference Servers",
        )

        return TritonServerSet(
            executor=triton_executor,
            triton_model_repository_path=paths.container_triton_model_repository_path,
            num_nodes=variant.pipeline_parallel_size,
            gpus_per_server=variant.tensor_parallel_size,
            max_time_min=DEFAULT_BENCHMARK_TIME_MIN,
            enable_gpus_allocation=config["slurm"].get("enable_gpus_allocation", True),
            verbose=verbose,
            config_name=variant.extended_name,
        )

    def _init_perf_jobs_executor():
        dirs_to_mount = get_dirs_to_mount(
            paths=paths, additional_path_pairs=[(dataset_dir, dataset_dir)], triton_model_repository_readonly=False
        )
        slurm_common_parameters = get_common_slurm_parameters(
            cluster_config=config,
            dirs_to_mount=dirs_to_mount,
            container_image_type=ContainerImageType.TRAINING,
        )
        executor_ = PyxisExecutor(folder=paths.submit_dir_path)
        executor_.update_parameters(
            **slurm_common_parameters,
            time=DEFAULT_BENCHMARK_TIME_MIN,
            job_name=f"joc-bermuda:eval_{variant.extended_name}",
            comment="Task for profiling of models",
            setup=["export NO_COLOR=1", f"export PYTHONPATH={sys.path[0]}"],
        )
        if config["slurm"].get("enable_gpus_allocation", True):
            executor_.update_parameters(
                gpus_per_node=8,
            )
        return executor_

    try:
        triton_server_set = _init_triton_set()
        triton_server_set_job = triton_server_set._job
        job_name = triton_server_set_job.get_info().get("JobName")
        LOGGER.info(
            f"[{triton_server_set_job.job_id}/{job_name}] Submitted set of Triton Servers "
            f"for Triton Model Repository {paths.host_triton_model_repository_path}"
        )
        LOGGER.info(f"[{triton_server_set_job.job_id}/{job_name}] logs: {triton_server_set_job.paths.stdout}")
        triton_server_set.wait_job_is_running_or_failed()
        if triton_server_set.failed_or_missing:
            LOGGER.info(
                f"[{triton_server_set_job.job_id}/{job_name}] Stopping benchmarking this config "
                f"(job state: {triton_server_set.state})"
            )
        else:
            executor = _init_perf_jobs_executor()
            perf_job = executor.submit(
                run_perf_test,
                config_path=paths.navigator_config_path,
                workspace_path=paths.workspace_path,
                bignlp_scripts_path=paths.host_cwd,
                triton_endpoint_url=triton_server_set.grpc_endpoints[0],
                dataset_dir=dataset_dir.resolve().absolute(),
                variant=variant,
                verbose=verbose,
            )
            job_name = perf_job.get_info().get("JobName")
            LOGGER.info(
                f"[{perf_job.job_id}/{job_name}] Submitted profiling job "
                f"for Triton Model Repository {paths.host_triton_model_repository_path}"
            )
            LOGGER.info(f"[{perf_job.job_id}/{job_name}] logs: {perf_job.paths.stdout}")

            perf_job.cancel_at_deletion()
            perf_job_results = perf_job.result()
            LOGGER.info(f"[{perf_job.job_id}/{job_name}] Evaluation results: {perf_job_results}")
    except submitit.core.utils.UncompletedJobError as e:
        LOGGER.warning(str(e))
    finally:
        if perf_job is not None:
            perf_job.cancel()
        if triton_server_set is not None:
            triton_server_set.stop()


def main():
    parser = _prepare_args_parser()
    args = parser.parse_args()

    config_logger(args.verbose)
    monkeypatch_submitit()

    cluster_config_path = pathlib.Path(args.cluster_config_path).resolve().absolute()
    with cluster_config_path.open("r") as config_file:
        cluster_config = yaml.load(config_file, Loader=yaml.SafeLoader)

    paths = Paths(
        host_cwd=pathlib.Path.cwd(),
        workspace_path=pathlib.Path(args.workspace_path).resolve().absolute(),
        original_model_path=pathlib.Path(args.model_path).resolve().absolute(),
        converted_model_path=None,
        navigator_config_path=pathlib.Path(args.navigator_config_path).resolve().absolute(),
        host_triton_model_repository_path=None,
        container_workdir=pathlib.Path(cluster_config["env"]["pyxis_container_workdir"]).resolve().absolute(),
        container_triton_model_repository_path=pathlib.Path(TRITON_MODEL_REPOSITORY),
    )

    for name, value in vars(args).items():
        LOGGER.info(f"{name}: {value}")
    for name, value in vars(paths).items():
        LOGGER.info(f"{name}: {value}")

    converted_model_path = _convert_model(
        paths=paths,
        config=cluster_config,
        model_name=args.model_name,
        verbose=args.verbose,
    )
    converted_paths = dataclasses.replace(paths, converted_model_path=converted_model_path)
    triton_model_repository_path = _prepare_triton_model_repository(
        paths=converted_paths,
        config=cluster_config,
        model_name=args.model_name,
        verbose=args.verbose,
    )
    deployed_paths = dataclasses.replace(
        converted_paths, host_triton_model_repository_path=triton_model_repository_path
    )

    _load_triton_model_and_run_benchmark(
        paths=deployed_paths,
        config=cluster_config,
        dataset_dir=pathlib.Path(args.dataset_dir).resolve().absolute(),
        verbose=args.verbose,
    )


def _prepare_args_parser():
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Test BigNLP models")
    parser.add_argument("--cluster-config-path", help="Path to cluster configuration file", required=True)
    parser.add_argument(
        "--navigator-config-path", help="Path to Triton Model Navigator configuration file", required=True
    )
    parser.add_argument(
        "--workspace-path",
        help="Path to workspace dir where logs and artifacts will be stored",
        default=f"./infer_workspace-{dt}",
    )
    parser.add_argument(
        "--model-path", help="Path to model checkpoint which will be converted and profiled", required=True
    )
    parser.add_argument("--model-name", help="Name of the model visible in Triton Inference Server", required=True)
    parser.add_argument(
        "--dataset-dir",
        help="Path to directory containing LAMBADA dataset and vocabulary files used for accuracy verification",
        required=True,
    )
    parser.add_argument("--verbose", "-v", help="Provides verbose output", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    main()
