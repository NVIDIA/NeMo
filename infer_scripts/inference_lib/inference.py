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
import enum
import logging
import pathlib
import shutil
import typing

import yaml

from .triton import Variant
from .utils import H2MIN, MIN2S

LOGGER = logging.getLogger(__name__)

DEFAULT_JOB_NAME_PREFIX = "bignlp-"
DEFAULT_MAX_CONFIG_TIME_MIN = 2 * H2MIN
DEFAULT_BENCHMARK_TIME_MIN = 2 * H2MIN
DEFAULT_MAX_WAIT_FOR_JOB_SCHEDULE_S = 30 * MIN2S

BIGNLP_SCRIPTS_PATH = pathlib.Path("/opt/bignlp/bignlp-scripts")
MA_PATCH_SCRIPT_REL_PATH = "infer_scripts/inference_lib/patches/patch_ma.py"


class ContainerImageType(enum.Enum):
    TRAINING = "training"
    INFERENCE = "inference"


INFERENCE_OUTPUT_FIELDS = [
    "model_name",
    "batch_size",
    "concurrency",
    "model_config_path",
    "backend_parameters",
    "dynamic_batch_sizes",
    "satisfies_constraints",
    "perf_throughput",
    "perf_throughput_normalized",
    "perf_latency_p50",
    "perf_latency_p90",
    "perf_latency_p95",
    "perf_latency_p99",
]


def get_triton_config_model_cmds(
    variant: Variant,
    workspace_path: pathlib.Path,
    navigator_config_path: pathlib.Path,
    src_model_path: pathlib.Path,
    triton_model_repository_path: pathlib.Path,
    verbose: bool = False,
):
    backend_parameters = {}
    if variant.is_half:
        backend_parameters["is_half"] = int(variant.is_half)
    if variant.pipeline_parallel_size:
        backend_parameters["pipeline_para_size"] = int(variant.pipeline_parallel_size)
    if variant.input_output_lengths_pair:
        input_length, output_length = variant.input_output_lengths_pair
        backend_parameters["max_input_len"] = input_length
        backend_parameters["max_seq_len"] = input_length + output_length

    parameters = []
    if variant.max_batch_size:
        parameters.extend(
            ["--max-batch-size", variant.max_batch_size, "--preferred-batch-sizes", variant.max_batch_size]
        )
    if backend_parameters:
        parameters.extend(
            ["--triton-backend-parameters", *[f"{name}={value}" for name, value in backend_parameters.items()]]
        )

    return [
        f"model-navigator triton-config-model "
        f"--workspace-path {workspace_path.resolve().absolute()} "
        f"--config-path {navigator_config_path} "
        f"--model-path {src_model_path.resolve().absolute()} "
        f"--model-name {variant.extended_name} "
        f"--model-repository {triton_model_repository_path.resolve().absolute()} "
        f"--use-symlinks "
        f"{'--verbose ' if verbose else ''}" + " ".join(map(str, parameters))
    ]


def get_convert_model_cmds(
    workspace_path: pathlib.Path,
    navigator_config_path: pathlib.Path,
    model_name: str,
    src_model_path: pathlib.Path,
    output_model_path: pathlib.Path,
    verbose: bool = False,
    tensor_parallel_size: typing.Optional[int] = None,
):
    if src_model_path.name.endswith(".ft"):
        return []
    else:
        # if need to overwrite tensor_parallel_size from navigator config path
        if tensor_parallel_size is not None:
            ft_gpu_sizes_arg = f"--ft-gpu-counts {tensor_parallel_size}"
        else:
            ft_gpu_sizes_arg = ""

        return [
            f"MODEL_NAVIGATOR_RUN_BY=1 model-navigator convert "
            f"--workspace-path {workspace_path.resolve().absolute()} "
            f"--config-path {navigator_config_path.resolve().absolute()} "
            f"--model-name {model_name} "
            f"--model-path {src_model_path.resolve().absolute()} "
            f"{ft_gpu_sizes_arg} "
            f"{'--model-format megatron ' if not src_model_path.suffix else ''}"
            f"--output-path {output_model_path.resolve().absolute()} "
            f"{'--verbose ' if verbose else ''}"
            f"--use-symlinks",
        ]


def convert_random_ft2ft(src_model_path, output_model_path, ft_gpu_count):
    if output_model_path.exists():
        raise RuntimeError(f"{output_model_path} already exists")
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_model_path, output_model_path)
    if ft_gpu_count is not None:

        def _rewrite_meta_file(ft_gpu_count_, output_model_path_):
            meta_path = output_model_path_ / "meta.yaml"
            with meta_path.open("r") as meta_file:
                meta_info = yaml.load(meta_file, Loader=yaml.SafeLoader)
            meta_info["tensor_para_size"] = int(ft_gpu_count_)
            with meta_path.open("w") as meta_file:
                yaml.dump(meta_info, meta_file)

        _rewrite_meta_file(ft_gpu_count, output_model_path)

    config = {}
    config_path = output_model_path.with_suffix(".nav.yaml")
    if config_path.exists():
        with config_path.open("r") as config_file:
            config = yaml.load(config_file, Loader=yaml.SafeLoader)

    config_needs_update = config.get("launch_mode") != "local" or (
        ft_gpu_count is not None and config.get("ft_gpu_counts") != [ft_gpu_count]
    )

    if config_needs_update:
        config["launch_mode"] = "local"
        config["ft_gpu_counts"] = [ft_gpu_count]

        with config_path.open("w") as config_file:
            yaml.dump(config, config_file)


def get_profile_cmds(
    *,
    workspace_path: pathlib.Path,
    navigator_config_path: pathlib.Path,
    model_repository_path: pathlib.Path,
    triton_endpoint_url: str,
    triton_metrics_url: str,
    verbose: bool = False,
):
    variant = Variant.from_triton_model_repository(model_repository_path)

    input_len, output_len = variant.input_output_lengths_pair

    max_shapes = [f"INPUT_ID=-1,1,{input_len}", "REQUEST_INPUT_LEN=-1,1", "REQUEST_OUTPUT_LEN=-1,1"]
    value_ranges = [
        f"INPUT_ID={variant.end_id},{variant.end_id}",
        f"REQUEST_INPUT_LEN={input_len},{input_len}",
        f"REQUEST_OUTPUT_LEN={output_len},{output_len}",
    ]
    dtypes = ["INPUT_ID=uint32", "REQUEST_INPUT_LEN=uint32", "REQUEST_OUTPUT_LEN=uint32"]
    # WAR for bug in perf_analyzer which fails if count search window is used
    # and if there are batch_sizes higher than max_batch_size
    with navigator_config_path.open("r") as config_file:
        nav_config = yaml.load(config_file, Loader=yaml.SafeLoader)
        config_search_batch_sizes = [
            bs for bs in nav_config.get("config_search_batch_sizes", []) if bs <= variant.max_batch_size
        ]
        config_search_batch_sizes_arg = (
            f"--config-search-batch-sizes {' '.join(map(str, config_search_batch_sizes))} "
            if config_search_batch_sizes
            else ""
        )

    return [
        f"python3 ${{BIGNLP_SCRIPTS_PATH}}/{MA_PATCH_SCRIPT_REL_PATH}",
        "model-navigator profile "
        f"--workspace-path {workspace_path} "
        f"--config-path {navigator_config_path} "
        f"--triton-endpoint-url {triton_endpoint_url} "
        f"--triton-metrics-url {triton_metrics_url} "
        f"--model-repository {model_repository_path} "
        f"--max-shapes {' '.join(max_shapes)} "
        f"--value-ranges {' '.join(value_ranges)} "
        f"--dtypes {' '.join(dtypes)} "
        f"{'--verbose ' if verbose else ''}" + config_search_batch_sizes_arg,
    ]
