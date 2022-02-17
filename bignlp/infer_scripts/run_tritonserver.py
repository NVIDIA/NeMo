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

import yaml

from inference_lib.cluster.executor import ClusterExecutor
from inference_lib.cluster.job import JobDefinition
from inference_lib.inference import DEFAULT_BENCHMARK_TIME_MIN
from inference_lib.triton import DEFAULT_GRPC_PORT, DEFAULT_HTTP_PORT, DEFAULT_METRIC_PORT, TritonServerSet, Variant
from inference_lib.utils import CLUSTER_DIR_NAME, MIN2S, config_logger

LOGGER = logging.getLogger("run_tritonserver")


def main():
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Test BigNLP models")
    parser.add_argument("--cluster-config-path", help="Path to cluster configuration file", required=True)
    parser.add_argument("--model-repository-path", help="Path to Triton Model Repository", required=True)
    parser.add_argument(
        "--workspace-path",
        help="Path to workspace dir where logs and artifacts will be stored",
        default=f"./infer_workspace-{dt}",
    )
    parser.add_argument("--verbose", "-v", help="Provides verbose output", action="store_true", default=False)
    args = parser.parse_args()

    workspace_path = pathlib.Path(args.workspace_path).resolve().absolute()
    config_logger(workspace_path, args.verbose)

    LOGGER.info(f"Arguments:")
    for name, value in vars(args).items():
        LOGGER.info(f"    {name}: {value}")

    config_path = pathlib.Path(args.cluster_config_path).resolve().absolute()
    with config_path.open("r") as config_file:
        cluster_config = yaml.load(config_file, Loader=yaml.SafeLoader)

    triton_model_repository_path = pathlib.Path(args.model_repository_path).resolve().absolute()
    cluster_dir_path = workspace_path / CLUSTER_DIR_NAME

    job_name_prefix = cluster_config["env"]["job_name_prefix"]
    inference_container_image = cluster_config["env"]["inference_container_image"]

    variant = Variant.from_triton_model_repository(triton_model_repository_path)
    LOGGER.info(f"Config variant {variant}")

    executor = ClusterExecutor(cluster_dir_path=cluster_dir_path, cluster_config=cluster_config["cluster"])

    tritonserver_set_job_def = JobDefinition(
        name=f"{job_name_prefix}tritonserver_set_{variant.extended_name}",
        description=f"Run Triton Inference Server for {triton_model_repository_path}",
        max_time_s=DEFAULT_BENCHMARK_TIME_MIN * MIN2S,
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
        raise RuntimeError(f"Failed job with set of Triton Inference Servers. state: {triton_server_set.state}")

    LOGGER.info(f"Triton Inference Server job has started")
    LOGGER.info(f"Triton Inference Server http endpoint url: {triton_server_set.http_endpoints[0]}")
    LOGGER.info(f"Triton Inference Server grpc endpoint url: {triton_server_set.grpc_endpoints[0]}")
    LOGGER.info(f"Server logs: {triton_server_set_job.log_path}")

    triton_server_set_job.wait()


if __name__ == "__main__":
    main()
