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

from inference_lib.inference import DEFAULT_BENCHMARK_TIME_MIN, TritonServerSet, Variant
from inference_lib.slurm import (
    DEFAULT_JOB_NAME_PREFIX,
    TRITON_MODEL_REPOSITORY,
    ContainerImageType,
    PyxisTritonExecutor,
    get_common_slurm_parameters_new,
)
from inference_lib.utils import config_logger, monkeypatch_submitit

LOGGER = logging.getLogger("run_tritonserver")


def _get_variant(host_triton_model_repository_path):
    triton_config_paths = list(host_triton_model_repository_path.rglob("config.pbtxt"))
    if len(triton_config_paths) > 1:
        raise ValueError(f"More than single config.pbtxt in {host_triton_model_repository_path}")
    elif not triton_config_paths:
        raise ValueError(f"Could not find config.pbtxt in {host_triton_model_repository_path}")
    triton_config_path = triton_config_paths[0]
    variant = Variant.from_triton_config(triton_config_path)
    LOGGER.info(f"Config variant {variant}")
    return variant


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

    config_logger(args.verbose)
    monkeypatch_submitit()

    for name, value in vars(args).items():
        LOGGER.info(f"{name}: {value}")

    config_path = pathlib.Path(args.cluster_config_path).resolve().absolute()
    with config_path.open("r") as config_file:
        cluster_config = yaml.load(config_file, Loader=yaml.SafeLoader)

    host_triton_model_repository_path = pathlib.Path(args.model_repository_path)
    container_triton_model_repository_path = pathlib.Path(TRITON_MODEL_REPOSITORY)
    workspace_path = pathlib.Path(args.workspace_path)
    slurm_workspace = workspace_path / "slurm_workspace"
    container_workdir_path = pathlib.Path(cluster_config["env"]["pyxis_container_workdir"])

    variant = _get_variant(host_triton_model_repository_path)

    job_name_prefix = cluster_config["slurm"].get("job_name_prefix", DEFAULT_JOB_NAME_PREFIX)
    enable_gpus_allocation = cluster_config["slurm"].get("enable_gpus_allocation", True)
    job_name = f"{job_name_prefix}tritonservers_{variant.extended_name}"

    slurm_common_parameters = get_common_slurm_parameters_new(
        cluster_config=cluster_config,
        dirs_to_mount=[
            (workspace_path, container_workdir_path),
            (host_triton_model_repository_path, container_triton_model_repository_path),
        ],
        container_image_type=ContainerImageType.INFERENCE,
    )
    triton_executor = PyxisTritonExecutor(folder=slurm_workspace)
    triton_executor.update_parameters(
        **slurm_common_parameters,
        time=DEFAULT_BENCHMARK_TIME_MIN,
        job_name=job_name,
        comment="Running set of Triton Inference Servers",
    )

    triton_server_set = TritonServerSet(
        executor=triton_executor,
        triton_model_repository_path=container_triton_model_repository_path,
        num_nodes=variant.pipeline_parallel_size,
        gpus_per_server=variant.tensor_parallel_size,
        enable_gpus_allocation=enable_gpus_allocation,
        max_time_min=DEFAULT_BENCHMARK_TIME_MIN,
        verbose=args.verbose,
        config_name=variant.extended_name,
        job_name_prefix=job_name_prefix,
    )

    triton_server_set.wait_job_is_running_or_failed()
    if triton_server_set.failed_or_missing:
        raise RuntimeError(f"Failed job with set of Triton Inference Servers. state: {triton_server_set.state}")

    LOGGER.info(f"Triton Inference Server is starting")
    LOGGER.info(f"Triton Inference Server grpc endpoint url: {triton_server_set.grpc_endpoints[0]}")
    LOGGER.info(f"Triton Inference Server http endpoint url: {triton_server_set.http_endpoints[0]}")
    LOGGER.info(f"Server logs: {triton_server_set._job.paths.stdout}")

    triton_server_set.wait_while_running()


if __name__ == "__main__":
    main()
