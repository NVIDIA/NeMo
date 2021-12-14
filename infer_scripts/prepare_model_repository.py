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
import datetime
import logging
import pathlib
import shutil

import yaml

from inference_lib.cluster.executor import ClusterExecutor
from inference_lib.cluster.job import JobDefinition
from inference_lib.inference import (
    BIGNLP_SCRIPTS_PATH,
    DEFAULT_MAX_CONFIG_TIME_MIN,
    get_convert_model_cmds,
    get_triton_config_model_cmds,
)
from inference_lib.triton import DEFAULT_GRPC_PORT, DEFAULT_HTTP_PORT, DEFAULT_METRIC_PORT, TritonServerSet, Variant
from inference_lib.utils import CLUSTER_DIR_NAME, FS_SYNC_TIMEOUT_S, MIN2S, config_logger, get_YN_input, wait_for

LOGGER = logging.getLogger("prepare_model_repo")

ACCURACY_REPORT_FILENAME = "lambada_metrics.csv"
OFFLINE_PERFORMANCE_REPORT_FILENAME = "triton_performance_offline.csv"
ONLINE_PERFORMANCE_REPORT_FILENAME = "triton_performance_online.csv"


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
    parser.add_argument("--model-repository-path", help="Path to result Triton Model Repository", required=True)
    parser.add_argument(
        "--dataset-dir",
        help="Path to directory containing LAMBADA dataset and vocabulary files used for accuracy verification",
        required=True,
    )
    parser.add_argument("--accuracy-tests", help="Run accuracy tests", action="store_true", default=False)
    parser.add_argument("--performance-tests", help="Run performance tests", action="store_true", default=False)
    parser.add_argument("--verbose", "-v", help="Provides verbose output", action="store_true", default=False)
    return parser


def main():
    parser = _prepare_args_parser()
    args = parser.parse_args()

    workspace_path = pathlib.Path(args.workspace_path).resolve().absolute()
    config_logger(workspace_path, args.verbose)

    LOGGER.info(f"Arguments:")
    for name, value in vars(args).items():
        LOGGER.info(f"  {name}: {value}")

    triton_model_repository_path = pathlib.Path(args.model_repository_path).resolve().absolute()
    if triton_model_repository_path.exists():
        delete_output_model_repository = get_YN_input(
            f"{triton_model_repository_path} exists. Do you want to remove it? [y/N] ", False
        )
        if delete_output_model_repository:
            LOGGER.info(f"Removing {triton_model_repository_path}")
            shutil.rmtree(triton_model_repository_path)
        else:
            LOGGER.warning(f"{triton_model_repository_path} exists.")
            return -1

    cluster_config_path = pathlib.Path(args.cluster_config_path).resolve().absolute()
    with cluster_config_path.open("r") as config_file:
        cluster_config = yaml.load(config_file, Loader=yaml.SafeLoader)

    src_model_path = pathlib.Path(args.model_path).resolve().absolute()
    navigator_config_path = pathlib.Path(args.navigator_config_path).resolve().absolute()
    cluster_dir_path = workspace_path / CLUSTER_DIR_NAME
    navigator_workspace_path = workspace_path / "navigator_workspace"

    job_name_prefix = cluster_config["env"]["job_name_prefix"]
    training_container_image = cluster_config["env"]["training_container_image"]
    inference_container_image = cluster_config["env"]["inference_container_image"]

    navigator_config_on_workspace_path = workspace_path / navigator_config_path.name
    navigator_config_on_workspace_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(navigator_config_path, navigator_config_on_workspace_path)

    executor = ClusterExecutor(cluster_dir_path=cluster_dir_path, cluster_config=cluster_config["cluster"])

    # convert
    variant = Variant(model_name=args.model_name)
    converted_model_path = workspace_path / f"{args.model_name}-converted.ft"
    triton_prepare_model_repository_job_def = JobDefinition(
        name=f"{job_name_prefix}prepare_model_repository",
        description=f"{src_model_path} model conversion and preparation of Triton Model Repository",
        max_time_s=DEFAULT_MAX_CONFIG_TIME_MIN * MIN2S,
        container_image=training_container_image,
        tasks_number=1,
        gpus_number_per_task=8,
        commands=[
            f"export BIGNLP_SCRIPTS_PATH={BIGNLP_SCRIPTS_PATH}",
            "export PYTHONPATH=${BIGNLP_SCRIPTS_PATH}:${PYTHONPATH}",
            "export PYTHONUNBUFFERED=1",
            "export NO_COLOR=1",
            *get_convert_model_cmds(
                workspace_path=navigator_workspace_path,
                navigator_config_path=navigator_config_on_workspace_path,
                model_name=args.model_name,
                src_model_path=src_model_path,
                output_model_path=converted_model_path,
                verbose=True,
            ),
            *get_triton_config_model_cmds(
                variant=variant,
                workspace_path=navigator_workspace_path,
                navigator_config_path=navigator_config_on_workspace_path,
                src_model_path=converted_model_path,
                triton_model_repository_path=triton_model_repository_path,
                verbose=True,
            ),
        ],
        directories_to_mount=[
            navigator_config_on_workspace_path.parent,
            src_model_path.parent,
            converted_model_path.parent,
            triton_model_repository_path.parent,
        ],
    )

    LOGGER.info(f"[-] Running job for {triton_prepare_model_repository_job_def.description}")
    triton_prepare_model_repository_job = executor.run(triton_prepare_model_repository_job_def)
    LOGGER.info(
        f"[{triton_prepare_model_repository_job.job_id}] " f"Triton Model Repository: {triton_model_repository_path}"
    )

    if any([args.accuracy_tests, args.performance_tests]):
        wait_for(
            "Triton Server Model configurations",
            predicate_fn=lambda: list(triton_model_repository_path.rglob("config.pbtxt")),
            timeout_s=FS_SYNC_TIMEOUT_S,
        )
        # run Triton server if tests requested
        variant = Variant.from_triton_model_repository(triton_model_repository_path)
        tritonserver_set_job_def = JobDefinition(
            name=f"{job_name_prefix}tritonserver_set_{variant.extended_name}",
            description=f"Run Triton Inference Server for {triton_model_repository_path}",
            max_time_s=DEFAULT_MAX_CONFIG_TIME_MIN * MIN2S,
            container_image=inference_container_image,
            commands=[
                f"export CUDA_VISIBLE_DEVICES={','.join(map(str, range(0, variant.tensor_parallel_size)))}",
                "export NCCL_LAUNCH_MODE=GROUP",
                f"tritonserver --model-repository {triton_model_repository_path} "
                f"{'--log-verbose 1' if args.verbose else ''}",
            ],
            directories_to_mount=[triton_model_repository_path],
            ports=[DEFAULT_HTTP_PORT, DEFAULT_GRPC_PORT, DEFAULT_METRIC_PORT],
            tasks_number=variant.pipeline_parallel_size,
            tasks_number_per_node=1,
            gpus_number_per_task=variant.tensor_parallel_size,
        )
        LOGGER.info(f"[-] Submitted job for {tritonserver_set_job_def.description}")
        triton_server_set_job = executor.submit(tritonserver_set_job_def)
        triton_server_set = TritonServerSet(triton_server_set_job)
        triton_server_set.wait_until_job_is_running_or_done()

        if triton_server_set.state.is_done():
            LOGGER.warning(
                f"[{triton_server_set_job.job_id}] Stopping benchmarking this config "
                f"(job state: {triton_server_set.state})"
            )
        else:
            server_url = triton_server_set.http_endpoints[0]

            accuracy_report_path = workspace_path / ACCURACY_REPORT_FILENAME
            offline_performance_report_path = workspace_path / OFFLINE_PERFORMANCE_REPORT_FILENAME
            online_performance_report_path = workspace_path / ONLINE_PERFORMANCE_REPORT_FILENAME
            commands = [
                f"export BIGNLP_SCRIPTS_PATH={BIGNLP_SCRIPTS_PATH}",
                "export PYTHONPATH=${BIGNLP_SCRIPTS_PATH}:${PYTHONPATH}",
                "export PYTHONUNBUFFERED=1",
                "export NO_COLOR=1",
            ]
            directories_to_mount = [navigator_config_on_workspace_path.parent]

            if args.accuracy_tests:
                protocol, host_and_port = server_url.split("://")
                dataset_dir = pathlib.Path(args.dataset_dir).resolve().absolute()
                batch_size = 128
                commands.append(
                    "${BIGNLP_SCRIPTS_PATH}/infer_scripts/evaluate_lambada.py "
                    f"-u {host_and_port} "
                    f"--protocol {protocol} "
                    f"-m {args.model_name} "
                    f"-d {dataset_dir} "
                    f"-b {batch_size} "
                    f"--output_csv {accuracy_report_path} "
                    "--n-gram-disabled "
                    f"{'--verbose' if args.verbose else ''}"
                )
                directories_to_mount.append(dataset_dir)

            if args.performance_tests:
                # WAR for bug in perf_analyzer which fails if count search window is used
                # and if there are batch_sizes higher than max_batch_size
                with navigator_config_on_workspace_path.open("r") as config_file:
                    config = yaml.load(config_file, Loader=yaml.SafeLoader)
                    max_batch_size = config.get("max_batch_size")
                    batch_sizes = [bs for bs in config.get("batch_sizes", []) if bs <= max_batch_size]
                    batch_sizes_arg = f"--batch-sizes {' '.join(map(str, batch_sizes))} " if batch_sizes else ""

                commands.extend(
                    [
                        f"model-navigator triton-evaluate-model "
                        f"--config-path {navigator_config_on_workspace_path} "
                        f"--server-url {server_url} "
                        f"--batching-mode static "
                        f"--evaluation-mode online "
                        f"--model-name {args.model_name} "
                        f"--latency-report-file {offline_performance_report_path} "
                        f"{'--verbose ' if args.verbose else ''}" + batch_sizes_arg,
                        f"model-navigator triton-evaluate-model "
                        f"--config-path {navigator_config_on_workspace_path} "
                        f"--server-url {server_url} "
                        f"--batching-mode dynamic "
                        f"--evaluation-mode online "
                        f"--model-name {args.model_name} "
                        f"--latency-report-file {online_performance_report_path} "
                        f"{'--verbose ' if args.verbose else ''}" + batch_sizes_arg,
                    ]
                )

            test_job_def = JobDefinition(
                name=f"{job_name_prefix}test_{args.model_name}",
                description=f"Testing of {server_url}/{args.model_name}",
                max_time_s=DEFAULT_MAX_CONFIG_TIME_MIN * MIN2S,
                container_image=training_container_image,
                commands=commands,
                directories_to_mount=directories_to_mount,
                tasks_number=1,
            )
            LOGGER.info(f"[-] Running job for {test_job_def.description}")
            test_job = executor.run(test_job_def, dependencies=[triton_server_set_job])

            if args.accuracy_tests:
                LOGGER.info(f"[{test_job.job_id}] Accuracy results: \n{accuracy_report_path.read_text()}")
            if args.performance_tests:
                LOGGER.info(
                    f"[{test_job.job_id}] "
                    f"Static batching benchmark results: \n{offline_performance_report_path.read_text()}"
                )
                LOGGER.info(
                    f"[{test_job.job_id}] "
                    f"Dynamic batching benchmark results: \n{online_performance_report_path.read_text()}"
                )


if __name__ == "__main__":
    main()
