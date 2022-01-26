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
import datetime
import itertools
import logging
import pathlib
import shutil
import sys
import textwrap

import yaml

from inference_lib.cluster.executor import ClusterExecutor
from inference_lib.cluster.job import JobDefinition
from inference_lib.inference import (
    BIGNLP_SCRIPTS_PATH,
    DEFAULT_BENCHMARK_TIME_MIN,
    DEFAULT_MAX_CONFIG_TIME_MIN,
    INFERENCE_OUTPUT_FIELDS,
    MA_PATCH_SCRIPT_REL_PATH,
    convert_random_ft2ft,
    get_convert_model_cmds,
    get_profile_cmds,
    get_triton_config_model_cmds,
)
from inference_lib.triton import DEFAULT_GRPC_PORT, DEFAULT_HTTP_PORT, DEFAULT_METRIC_PORT, TritonServerSet, Variant
from inference_lib.utils import CLUSTER_DIR_NAME, FS_SYNC_TIMEOUT_S, MIN2S, config_logger, wait_for

LOGGER = logging.getLogger("profile_model")


class InputOutputLength:
    def __init__(self, entry):
        input_, output_ = entry.split(",")
        self._input_length = int(input_)
        self._output_length = int(output_)

    def as_tuple(self):
        return self._input_length, self._output_length


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


def main():
    parser = _prepare_args_parser()
    args = parser.parse_args()
    args = _process_args(args)

    workspace_path = pathlib.Path(args.workspace_path).resolve().absolute()
    config_logger(workspace_path, args.verbose)

    LOGGER.info(f"Arguments:")
    for name, value in vars(args).items():
        LOGGER.info(f"  {name}: {value}")

    cluster_config_path = pathlib.Path(args.cluster_config_path).resolve().absolute()
    with cluster_config_path.open("r") as config_file:
        cluster_config = yaml.load(config_file, Loader=yaml.SafeLoader)

    src_model_path = pathlib.Path(args.model_path).resolve().absolute()
    navigator_config_path = pathlib.Path(args.navigator_config_path).resolve().absolute()
    cluster_dir_path = workspace_path / CLUSTER_DIR_NAME

    navigator_workspace_path = workspace_path / "navigator_workspace"
    interim_model_repository_path = navigator_workspace_path / "analyzer/interim-model-store"
    analyzer_model_repository_path = navigator_workspace_path / "analyzer/model-store"

    job_name_prefix = cluster_config["env"]["job_name_prefix"]
    training_container_image = cluster_config["env"]["training_container_image"]
    inference_container_image = cluster_config["env"]["inference_container_image"]

    navigator_config_on_workspace_path = workspace_path / navigator_config_path.name
    navigator_config_on_workspace_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(navigator_config_path, navigator_config_on_workspace_path)

    executor = ClusterExecutor(cluster_dir_path=cluster_dir_path, cluster_config=cluster_config["cluster"])

    is_halfs = ["1"]
    model_names = [args.model_name]
    for tensor_parallel_size in args.tensor_parallel_sizes:
        # convert and prepare model repositories
        converted_model_path = workspace_path / f"{args.model_name}-tp_{tensor_parallel_size}.ft"

        variants = list(
            map(
                lambda items: Variant(
                    model_name=items[0],
                    max_batch_size=items[1],
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=items[2],
                    input_output_lengths_pair=items[3],
                    is_half=items[4],
                ),
                itertools.product(
                    model_names, args.max_batch_sizes, args.pipeline_parallel_sizes, args.input_output_lengths, is_halfs
                ),
            )
        )
        triton_model_repositories_paths = [
            workspace_path / f"model_repo_{variant.extended_name}" for variant in variants
        ]

        is_ft2ft_conversion = src_model_path.name.endswith(".ft")
        if is_ft2ft_conversion:
            # convert FT checkpoint without weights locally - on head node/developer machine
            convert_random_ft2ft(src_model_path, converted_model_path, tensor_parallel_size)

        commands = [
            f"export BIGNLP_SCRIPTS_PATH={BIGNLP_SCRIPTS_PATH}",
            "export PYTHONPATH=${BIGNLP_SCRIPTS_PATH}:${PYTHONPATH}",
            "export PYTHONUNBUFFERED=1",
            "export NO_COLOR=1",
            *get_convert_model_cmds(
                workspace_path=navigator_workspace_path,
                navigator_config_path=navigator_config_on_workspace_path,
                tensor_parallel_size=tensor_parallel_size,
                model_name=args.model_name,
                src_model_path=src_model_path,
                output_model_path=converted_model_path,
                verbose=True,
            ),
        ] + [
            cmd
            for variant, triton_model_repository_path in zip(variants, triton_model_repositories_paths)
            for cmd in get_triton_config_model_cmds(
                variant=variant,
                workspace_path=navigator_workspace_path,
                navigator_config_path=navigator_config_on_workspace_path,
                src_model_path=converted_model_path,
                triton_model_repository_path=triton_model_repository_path,
                verbose=True,
            )
        ]
        triton_prepare_model_repository_job_def = JobDefinition(
            name=f"{job_name_prefix}prepare_model_repository-tp_{tensor_parallel_size}",
            description=(
                f"{src_model_path} model conversion and preparation of Triton Model Repository "
                f"for tensor parallel size {tensor_parallel_size}"
            ),
            max_time_s=DEFAULT_MAX_CONFIG_TIME_MIN * MIN2S,
            container_image=training_container_image,
            tasks_number=1,
            gpus_number_per_task=8,
            commands=commands,
            directories_to_mount=[
                navigator_config_on_workspace_path.parent,
                converted_model_path.parent,
                navigator_workspace_path,
                *[
                    triton_model_repositories_path.parent
                    for triton_model_repositories_path in triton_model_repositories_paths
                ],
            ]
            + ([] if is_ft2ft_conversion else [src_model_path.parent]),
        )
        LOGGER.info(f"[-] Running job for {triton_prepare_model_repository_job_def.description}")
        triton_prepare_model_repository_job = executor.run(triton_prepare_model_repository_job_def)

        wait_for(
            "Triton Server Model configurations",
            predicate_fn=lambda: all([list(p.rglob("config.pbtxt")) for p in triton_model_repositories_paths]),
            timeout_s=FS_SYNC_TIMEOUT_S,
        )

        triton_model_repositories_paths_block = textwrap.indent(
            "\n".join(
                triton_model_repository_path.as_posix()
                for triton_model_repository_path in triton_model_repositories_paths
            ),
            prefix="    - ",
        )
        LOGGER.info(
            f"[{triton_prepare_model_repository_job.job_id}] "
            f"Triton Model Repositories: \n{triton_model_repositories_paths_block}"
            f""
        )

        for variant, triton_model_repository_path in zip(variants, triton_model_repositories_paths):
            # start Triton server
            variant = Variant.from_triton_model_repository(triton_model_repository_path)
            tritonserver_set_job_def = JobDefinition(
                name=f"{job_name_prefix}tritonserver_set_{variant.model_name}",
                description=f"Triton Inference Server for {triton_model_repository_path}",
                max_time_s=DEFAULT_MAX_CONFIG_TIME_MIN * MIN2S,
                container_image=inference_container_image,
                commands=[
                    f"export CUDA_VISIBLE_DEVICES={','.join(map(str, range(0, variant.tensor_parallel_size)))}",
                    "export NCCL_LAUNCH_MODE=GROUP",
                    f"tritonserver --model-repository {triton_model_repository_path} "
                    f"{'--log-verbose 1' if args.verbose else ''}",
                ],
                directories_to_mount=[triton_model_repository_path.parent],
                ports=[DEFAULT_HTTP_PORT, DEFAULT_GRPC_PORT, DEFAULT_METRIC_PORT],
                tasks_number=variant.pipeline_parallel_size,
                tasks_number_per_node=1,
                gpus_number_per_task=variant.tensor_parallel_size,
            )

            LOGGER.info(f"[-] Submitting job for {tritonserver_set_job_def.description}")
            triton_server_set_job = executor.submit(tritonserver_set_job_def)
            triton_server_set = TritonServerSet(triton_server_set_job)
            try:
                triton_server_set.wait_until_job_is_running_or_done()

                if triton_server_set.state.is_done():
                    LOGGER.warning(
                        f"[{triton_server_set_job.job_id}] Stopping benchmarking this config "
                        f"(job state: {triton_server_set.state})"
                    )
                else:
                    # TODO: fix profiling over http protocol
                    # run profile
                    server_url = triton_server_set.grpc_endpoints[0]
                    metric_url = triton_server_set.metric_endpoints[0]

                    profile_job_def = JobDefinition(
                        name=f"{job_name_prefix}profile_{variant.model_name}",
                        description=f"Profile of {server_url}/{variant.model_name} model",
                        max_time_s=DEFAULT_BENCHMARK_TIME_MIN * MIN2S,
                        container_image=training_container_image,
                        commands=[
                            f"export BIGNLP_SCRIPTS_PATH={BIGNLP_SCRIPTS_PATH}",
                            "export PYTHONPATH=${BIGNLP_SCRIPTS_PATH}:${PYTHONPATH}",
                            "export PYTHONUNBUFFERED=1",
                            "export NO_COLOR=1",
                            *get_profile_cmds(
                                workspace_path=navigator_workspace_path,
                                navigator_config_path=navigator_config_on_workspace_path,
                                model_repository_path=triton_model_repository_path,
                                triton_endpoint_url=server_url,
                                triton_metrics_url=metric_url,
                                verbose=True,
                            ),
                            f"mkdir -p {interim_model_repository_path}",
                            f"find {triton_model_repository_path} "
                            f"-maxdepth 1 "
                            f"-type d "
                            f"-not -path {triton_model_repository_path} "
                            f"-exec sh -c 'ln -s {{}} {interim_model_repository_path}/$(basename {{}})' \\;",
                        ],
                        directories_to_mount=[
                            navigator_config_on_workspace_path.parent,
                            navigator_workspace_path,
                            triton_model_repository_path,
                        ],
                        tasks_number=1,
                    )
                    LOGGER.info(f"[-] Running job for {profile_job_def.description}")
                    profile_job = executor.run(profile_job_def, dependencies=[triton_server_set_job])
                    LOGGER.info(f"[{profile_job.job_id}] {profile_job.info.description} job finished")
            except Exception as e:
                LOGGER.warning(e)
            finally:
                triton_server_set_job.cancel()

    if list(interim_model_repository_path.glob("*")):
        analysis_summary_path = workspace_path / "analysis_summary.log"
        analyze_job_def = JobDefinition(
            name=f"{job_name_prefix}analyze",
            description=f"Analyze profile results",
            max_time_s=DEFAULT_MAX_CONFIG_TIME_MIN * MIN2S,
            container_image=training_container_image,
            commands=[
                f"export BIGNLP_SCRIPTS_PATH={BIGNLP_SCRIPTS_PATH}",
                "export PYTHONPATH=${BIGNLP_SCRIPTS_PATH}:${PYTHONPATH}",
                "export PYTHONUNBUFFERED=1",
                "export NO_COLOR=1",
                f"python3 ${{BIGNLP_SCRIPTS_PATH}}/{MA_PATCH_SCRIPT_REL_PATH}",
                f"rm -rf {analyzer_model_repository_path}",
                f"mv {interim_model_repository_path} {analyzer_model_repository_path}",
                "model-navigator analyze "
                f"--workspace-path {navigator_workspace_path} "
                f"--model-repository {analyzer_model_repository_path} "
                f"--top-n-configs {args.top_n_configs} "
                f"--max-latency-ms {args.max_latency_ms} "
                f"--inference-output-fields {' '.join(INFERENCE_OUTPUT_FIELDS)} "
                f"--objectives perf_throughput_normalized=10 "
                f"2>&1|tee {analysis_summary_path}",
            ],
            directories_to_mount=[
                navigator_config_on_workspace_path.parent,
                analyzer_model_repository_path.parent,
                analysis_summary_path.parent,
                navigator_workspace_path,
            ],
            tasks_number=1,
        )
        LOGGER.info(f"[-] Running job for {analyze_job_def.description}")
        analyze_job = executor.run(analyze_job_def)
        LOGGER.info(f"[{analyze_job.job_id}] summary: \n{analysis_summary_path.read_text()}")
    else:
        LOGGER.warning("No successful profile results were found, thus don't run analyze job.")
        sys.exit(1)


if __name__ == "__main__":
    main()
