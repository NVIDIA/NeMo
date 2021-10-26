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
import copy
import dataclasses
import datetime
import itertools
import logging
import pathlib
import sys
import typing

import submitit
import submitit.core
import yaml

from inference_lib.inference import (
    DEFAULT_BENCHMARK_TIME_MIN,
    DEFAULT_MAX_CONFIG_TIME_MIN,
    Paths,
    TritonServerSet,
    Variant,
    convert_model_ft2ft,
    get_dirs_to_mount,
    run_analyze,
    run_profile,
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

LOGGER = logging.getLogger("profile_model")


class InputOutputLength:
    def __init__(self, entry):
        input_, output_ = entry.split(",")
        self._input_length = int(input_)
        self._output_length = int(output_)

    def as_tuple(self):
        return self._input_length, self._output_length


def _convert_model(
    *,
    config,
    paths,
    model_name,
    tensor_parallel_size: int,
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
            job_name=f"joc-bermuda:convert_model-tp_{tensor_parallel_size}",
            comment="Task for converting Megatron/NeMo model to Fastertransformer format",
            setup=["export NO_COLOR=1", f"export PYTHONPATH={sys.path[0]}"],
        )
        return executor_

    try:
        executor = _init_executor()

        conversion_job = executor.submit(
            convert_model_ft2ft,
            config_path=paths.navigator_config_path,
            workspace_path=paths.workspace_path,
            model_name=model_name,
            model_path=paths.original_model_path,
            verbose=verbose,
            parameters_to_override={
                "ft_gpu_counts": tensor_parallel_size,
            },
        )
        setup_job(conversion_job)
        job_name = conversion_job.get_info().get("JobName")
        LOGGER.info(
            f"[{conversion_job.job_id}/{job_name}] Submitted conversion for model {paths.original_model_path} "
            f"with tensor_parallel_size: {tensor_parallel_size}"
        )
        LOGGER.info(f"[{conversion_job.job_id}/{job_name}] logs: {conversion_job.paths.stdout}")
        converted_model_path = conversion_job.result()
        LOGGER.info(f"[{conversion_job.job_id}/{job_name}] converted model: {converted_model_path}")
        return pathlib.Path(converted_model_path)
    finally:
        if conversion_job is not None:
            conversion_job.cancel()


def _prepare_triton_model_repositories(
    *,
    config,
    paths,
    variants: typing.List[Variant],
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
        return executor_

    try:
        executor = _init_executor()
        preparation_job = executor.submit(
            triton_config_model,
            config_path=paths.navigator_config_path,
            workspace_path=paths.workspace_path,
            model_path=paths.converted_model_path,
            variants=variants,
            verbose=verbose,
        )
        setup_job(preparation_job)
        job_name = preparation_job.get_info().get("JobName")
        LOGGER.info(
            f"[{preparation_job.job_id}/{job_name}] Submitted task for preparation set of Triton Model Repositories "
            f"for model {paths.converted_model_path}"
        )
        LOGGER.info(f"[{preparation_job.job_id}/{job_name}] logs: {preparation_job.paths.stdout}")

        model_repository_paths = preparation_job.result()
        LOGGER.info(f"[{preparation_job.job_id}/{job_name}] Triton Model Repositories:")
        for model_repository_path in model_repository_paths:
            LOGGER.info(f"[{preparation_job.job_id}/{job_name}]     - {model_repository_path}")

        return model_repository_paths
    finally:
        if preparation_job is not None:
            preparation_job.cancel()


def _load_triton_model_and_profile(
    *,
    paths: Paths,
    config: typing.Dict,
    config_name: str,
    variant: Variant,
    verbose: bool = False,
):
    triton_server_set = None
    perf_job = None

    def _init_triton_set():
        dirs_to_mount = get_dirs_to_mount(paths=paths, triton_model_repository_readonly=True)
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
            config_name=config_name,
        )

    def _init_profile_jobs_executor():
        dirs_to_mount = get_dirs_to_mount(paths=paths, triton_model_repository_readonly=False)
        slurm_common_parameters = get_common_slurm_parameters(
            cluster_config=config,
            dirs_to_mount=dirs_to_mount,
            container_image_type=ContainerImageType.TRAINING,
        )
        executor_ = PyxisExecutor(folder=paths.submit_dir_path)
        executor_.update_parameters(
            **slurm_common_parameters,
            time=DEFAULT_BENCHMARK_TIME_MIN,
            job_name=f"joc-bermuda:profile_{config_name}",
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
            executor = _init_profile_jobs_executor()
            perf_job = executor.submit(
                run_profile,
                paths=paths,
                config_path=paths.navigator_config_path,
                workspace_path=paths.workspace_path,
                model_repository_path=paths.container_triton_model_repository_path,
                triton_endpoint_url=triton_server_set.grpc_endpoints[0],
                triton_metrics_url=triton_server_set.metric_endpoints[0],
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
            LOGGER.info(f"[{perf_job.job_id}/{job_name}] Profiling results: {perf_job_results}")
    except submitit.core.utils.UncompletedJobError as e:
        LOGGER.warning(str(e))
    finally:
        if perf_job is not None:
            perf_job.cancel()
        if triton_server_set is not None:
            triton_server_set.stop()


def _analyze_results(
    *,
    paths: Paths,
    config: typing.Dict,
    top_n_configs: int,
    max_latency_ms: int,
    verbose: bool = False,
):
    analyze_job = None

    def _init_analyze_jobs_executor():
        dirs_to_mount = get_dirs_to_mount(paths=paths, triton_model_repository_readonly=False)
        slurm_common_parameters = get_common_slurm_parameters(
            cluster_config=config,
            dirs_to_mount=dirs_to_mount,
            container_image_type=ContainerImageType.TRAINING,
        )
        executor_ = PyxisExecutor(folder=paths.submit_dir_path)
        executor_.update_parameters(
            **slurm_common_parameters,
            time=DEFAULT_BENCHMARK_TIME_MIN,
            job_name=f"joc-bermuda:analysis",
            comment="Task for profiling results analysis",
            setup=["export NO_COLOR=1", f"export PYTHONPATH={sys.path[0]}"],
        )
        if config["slurm"].get("enable_gpus_allocation", True):
            executor_.update_parameters(
                gpus_per_node=8,
            )
        return executor_

    try:
        executor = _init_analyze_jobs_executor()
        analyze_job = executor.submit(
            run_analyze,
            paths=paths,
            workspace_path=paths.workspace_path,
            top_n_configs=top_n_configs,
            max_latency_ms=max_latency_ms,
            verbose=verbose,
        )
        analyze_job.cancel_at_deletion()
        job_name = analyze_job.get_info().get("JobName")
        LOGGER.info(
            f"[{analyze_job.job_id}/{job_name}] Submitted analysis job "
            f"for Triton Model Repository {paths.host_triton_model_repository_path}"
        )
        LOGGER.info(f"[{analyze_job.job_id}/{job_name}] logs: {analyze_job.paths.stdout}")
        analyze_job_results = analyze_job.result()
        LOGGER.info(f"[{analyze_job.job_id}/{job_name}] Analysis results: {analyze_job_results}")
    finally:
        if analyze_job is not None:
            analyze_job.cancel()


def main():
    parser = _prepare_args_parser()
    args = parser.parse_args()
    args = _process_args(args)

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

    is_halfs = ["1"]
    model_names = [args.model_name]
    for tensor_parallel in args.tensor_parallel_sizes:
        converted_model_path = _convert_model(
            paths=paths,
            config=cluster_config,
            model_name=args.model_name,
            tensor_parallel_size=tensor_parallel,
            verbose=args.verbose,
        )
        converted_paths = dataclasses.replace(paths, converted_model_path=converted_model_path)

        variants = list(
            map(
                lambda items: Variant(
                    model_name=items[0],
                    max_batch_size=items[1],
                    tensor_parallel_size=tensor_parallel,
                    pipeline_parallel_size=items[2],
                    input_output_lengths_pair=items[3],
                    is_half=items[4],
                ),
                itertools.product(
                    model_names, args.max_batch_sizes, args.pipeline_parallel_sizes, args.input_output_lengths, is_halfs
                ),
            )
        )
        triton_model_repository_paths = _prepare_triton_model_repositories(
            paths=converted_paths,
            config=cluster_config,
            variants=variants,
            verbose=args.verbose,
        )
        for variant, triton_model_repository_path in zip(variants, triton_model_repository_paths):
            profile_paths = dataclasses.replace(
                converted_paths,
                host_triton_model_repository_path=triton_model_repository_path,
            )
            _load_triton_model_and_profile(
                paths=profile_paths,
                config=cluster_config,
                config_name=variant.extended_name,
                variant=variant,
                verbose=args.verbose,
            )

    _analyze_results(
        paths=paths,
        config=cluster_config,
        top_n_configs=args.top_n_configs,
        max_latency_ms=args.max_latency_ms,
        verbose=args.verbose,
    )


def _process_args(args):
    def _flatten_list_of_list(list_of_list):
        # required because action="append" + nargs="+" are used in some arguments which creates list of list
        return [item for items in list_of_list for item in items]

    updated_args = copy.copy(args)
    updated_args.max_batch_sizes = _flatten_list_of_list(args.max_batch_sizes)
    updated_args.tensor_parallel_sizes = _flatten_list_of_list(args.tensor_parallel_sizes)
    updated_args.pipeline_parallel_sizes = _flatten_list_of_list(args.pipeline_parallel_sizes)
    updated_args.input_output_lengths = [item.as_tuple() for item in _flatten_list_of_list(args.input_output_lengths)]

    return updated_args


def _prepare_args_parser():
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Profile BigNLP models")
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
        "--max-batch-sizes",
        help="List of max_batch_sizes of the Triton Inference Server",
        action="append",
        nargs="+",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--tensor-parallel-sizes",
        help="Sizes of tensor parallel",
        action="append",
        nargs="+",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--pipeline-parallel-sizes",
        help="Sizes of pipeline parallel",
        action="append",
        nargs="+",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--input-output-lengths",
        help="List of input, output lengths pairs. Format: input_len,output_len [input_len,output_len ...]",
        action="append",
        nargs="+",
        default=[],
        type=InputOutputLength,
        required=True,
    )
    parser.add_argument(
        "--max-latency-ms", help="Maximum latency in ms that the analyzed models should match.", type=int, default=60000
    )
    parser.add_argument(
        "--top-n-configs", help="Number of top final configurations selected from the analysis.", type=int, default=10
    )
    parser.add_argument("--verbose", "-v", help="Provides verbose output", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    main()
