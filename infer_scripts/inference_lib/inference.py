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
import dataclasses
import logging
import math
import os
import pathlib
import re
import shutil
import site
import subprocess
import time
import typing

import submitit

from .slurm import (
    TRITON_MODEL_REPOSITORY,
    DirToMount,
    PyxisTritonExecutor,
    get_cluster_suffix,
    init_job_env,
    parse_node_list,
    setup_job,
)
from .utils import H2MIN, MIN2S

LOGGER = logging.getLogger(__name__)
DEFAULT_HTTP_PORT = 8000
DEFAULT_GRPC_PORT = 8001
DEFAULT_METRIC_PORT = 8002
DEFAULT_MAX_CONFIG_TIME_MIN = 2 * H2MIN
DEFAULT_BENCHMARK_TIME_MIN = 2 * H2MIN
DEFAULT_MAX_WAIT_FOR_JOB_SCHEDULE_S = 30 * MIN2S


@dataclasses.dataclass
class Paths:
    host_cwd: pathlib.Path
    workspace_path: pathlib.Path
    original_model_path: pathlib.Path
    converted_model_path: typing.Optional[pathlib.Path]
    navigator_config_path: pathlib.Path
    host_triton_model_repository_path: typing.Optional[pathlib.Path]
    container_workdir: pathlib.Path
    container_triton_model_repository_path: pathlib.Path = pathlib.Path(TRITON_MODEL_REPOSITORY)

    @property
    def submit_dir_path(self):
        return self.workspace_path / "slurm_workspace"


@dataclasses.dataclass
class Variant:
    model_name: str
    max_batch_size: typing.Optional[int] = None
    tensor_parallel_size: typing.Optional[int] = None
    pipeline_parallel_size: typing.Optional[int] = None
    input_output_lengths_pair: typing.Optional[typing.Tuple[int, int]] = None
    is_half: typing.Optional[bool] = None

    @property
    def extended_name(self):
        io = (
            f"{self.input_output_lengths_pair[0]}_{self.input_output_lengths_pair[1]}"
            if self.input_output_lengths_pair
            else None
        )
        parameters_to_be_included_in_name = [
            ("mbs", self.max_batch_size),
            ("pp", self.pipeline_parallel_size),
            ("tp", self.tensor_parallel_size),
            ("half", self.is_half),
            ("io", io),
        ]
        parameters = "-".join([f"{k}_{v}" for k, v in parameters_to_be_included_in_name if v is not None])
        if parameters:
            extended_name = f"{self.model_name}-{parameters}"
        else:
            extended_name = self.model_name
        return extended_name

    @classmethod
    def from_triton_config(cls, triton_config_path: pathlib.Path):
        triton_config_content = triton_config_path.read_text()

        def _search_backend_param(key: str) -> typing.Optional[str]:
            pattern = rf'{key}.*\n.*\n.*string_value:.*"(?P<value>.*)"'
            match = re.search(pattern, triton_config_content)
            if match is None:
                LOGGER.warning(f"Could not find {pattern} in {triton_config_content}")
                return None
            else:
                return match.groupdict()["value"]

        def _get_param(key: str):
            pattern = rf"{key}: (?P<value>.*)"
            match = re.search(pattern, triton_config_content)
            if match is None:
                LOGGER.warning(f"Could not find {pattern} in {triton_config_content}")
                return None
            else:
                value = match.groupdict()["value"]
                value = value.strip('"')
                return value

        def _cast_if_not_none(value, type_):
            return value if value is None else type_(value)

        pipeline_para_size = _search_backend_param("pipeline_para_size")
        tensor_para_size = _search_backend_param("tensor_para_size")
        is_half = _search_backend_param("is_half")
        input_len = _search_backend_param("max_input_len")
        seq_len = _search_backend_param("max_seq_len")
        if input_len is not None and seq_len is not None:
            output_len = int(seq_len) - int(input_len)
        else:
            output_len = None
        max_batch_size = _get_param("max_batch_size")

        return cls(
            model_name=_get_param("name"),
            max_batch_size=_cast_if_not_none(max_batch_size, int),
            tensor_parallel_size=_cast_if_not_none(tensor_para_size, int),
            pipeline_parallel_size=_cast_if_not_none(pipeline_para_size, int),
            input_output_lengths_pair=None if input_len is None or output_len is None else (input_len, output_len),
            is_half=_cast_if_not_none(is_half, bool),
        )


class TritonServerSet:
    def __init__(
        self,
        *,
        executor: PyxisTritonExecutor,
        triton_model_repository_path: pathlib.Path,
        config_name: str,
        num_nodes: int = 1,
        servers_per_node: int = 1,
        gpus_per_server: int = 8,
        enable_gpus_allocation: bool = True,
        max_time_min: int,
        verbose: bool = False,
    ):
        self._job = self._submit(
            executor=executor,
            repository_path=triton_model_repository_path,
            num_nodes=num_nodes,
            servers_per_node=servers_per_node,
            gpus_per_server=gpus_per_server,
            enable_gpus_allocation=enable_gpus_allocation,
            max_time_min=max_time_min,
            verbose=verbose,
            config_name=config_name,
        )
        self._run_since_s = math.inf

    def _submit(
        self,
        *,
        executor,
        repository_path,
        num_nodes,
        servers_per_node,
        gpus_per_server,
        enable_gpus_allocation,
        max_time_min,
        verbose,
        config_name,
    ) -> submitit.Job:
        executor.update_parameters(
            nodes=num_nodes,
            ntasks_per_node=servers_per_node,
            exclusive=True,
            time=max_time_min,
            job_name=f"joc-bermuda:tritonserver_set_{config_name}",
            comment=f"Triton Server serving {repository_path}",
            setup=[
                f"export CUDA_VISIBLE_DEVICES={','.join(map(str, range(0, gpus_per_server)))}",
                "export NCCL_LAUNCH_MODE=GROUP",
            ],
        )
        if enable_gpus_allocation:
            executor.update_parameters(
                gpus_per_node=gpus_per_server,
            )

        def _dummy_fn():
            pass

        job = executor.submit(_dummy_fn)
        setup_job(job)
        return job

    @property
    def nodes(self):
        info = self._job.get_info()
        return parse_node_list(info["NodeList"])

    @property
    def grpc_endpoints(self) -> typing.List[str]:
        return [f"grpc://{node}:{DEFAULT_GRPC_PORT}" for node in self.nodes]

    @property
    def http_endpoints(self) -> typing.List[str]:
        return [f"http://{node}:{DEFAULT_HTTP_PORT}" for node in self.nodes]

    @property
    def metric_endpoints(self) -> typing.List[str]:
        return [f"http://{node}:{DEFAULT_METRIC_PORT}/metrics" for node in self.nodes]

    def stop(self):
        self._job.cancel()

    @property
    def job(self):
        return self._job

    def wait_job_is_running_or_failed(self):
        timeout_s = DEFAULT_MAX_WAIT_FOR_JOB_SCHEDULE_S
        step_s = 5
        while not self.running and not self.failed_or_missing and timeout_s > 0:
            time.sleep(step_s)
            self._job.watcher.update()
            timeout_s -= step_s

        if timeout_s <= 0:
            raise TimeoutError(f"[{self._job.job_id}] Could not schedule TritonServerSet job")

    @property
    def state(self):
        info = self._job.get_info()
        return info.get("State", "UNKNOWN")

    @property
    def running(self):
        state_ = self.state
        return state_ in ["RUNNING"]

    @property
    def failed_or_missing(self):
        state_ = self.state
        result_ = state_ not in ["PENDING", "RUNNING", "UNKNOWN"]
        if result_:
            LOGGER.debug(f"Current job state {state_}")
        return result_

    def wait_while_running(self):
        self._job.wait()


def get_dirs_to_mount_new(path_pairs: typing.List[typing.Tuple[pathlib.Path, pathlib.Path]]):
    entries = [
        DirToMount(src.resolve().absolute(), dst.resolve().absolute(), readonly=False) for src, dst in path_pairs
    ]

    for entry in entries:
        if not entry.readonly and entry.src and not entry.src.exists():
            LOGGER.debug(f"Creating {entry.src}")
            entry.src.mkdir(parents=True, exist_ok=True)

    entries = [entry for entry in entries if entry.src]

    host_model_navigator_path = os.environ.get("MODEL_NAVIGATOR_DIR")
    if host_model_navigator_path:
        host_model_navigator_path = pathlib.Path(host_model_navigator_path).resolve().absolute()
        if host_model_navigator_path.exists():
            LOGGER.warning(f"Using Model Navigator mounted sources: {host_model_navigator_path}")
            container_model_navigator_path = pathlib.Path("/workspace/model_navigator")
            entries += [DirToMount(host_model_navigator_path, container_model_navigator_path, readonly=True)]

    return entries


def get_dirs_to_mount(
    *,
    paths: Paths,
    additional_path_pairs: typing.Optional[typing.List[typing.Tuple[pathlib.Path, pathlib.Path]]] = None,
    triton_model_repository_readonly: bool = True,
):
    additional_path_pairs = additional_path_pairs or []
    entries = [
        # to be able to store files in pyxis_container_workdir
        DirToMount(paths.workspace_path, paths.workspace_path, readonly=False),
        DirToMount(paths.workspace_path, paths.container_workdir, readonly=False),
        # to be able to load files from bignlp-scripts repository
        DirToMount(pathlib.Path.cwd(), pathlib.Path.cwd(), readonly=False),
        # to be able to load model checkpoint
        DirToMount(paths.original_model_path, paths.original_model_path, readonly=True),
        # TODO: fix navigator to be able to operate on readonly fs
        DirToMount(paths.navigator_config_path.parent, paths.navigator_config_path.parent, readonly=False),
        DirToMount(
            paths.host_triton_model_repository_path,
            pathlib.Path(TRITON_MODEL_REPOSITORY),
            readonly=triton_model_repository_readonly,
        ),
    ] + [
        DirToMount(src.resolve().absolute(), dst.resolve().absolute(), readonly=False)
        for src, dst in additional_path_pairs
    ]

    for entry in entries:
        if not entry.readonly and entry.src and not entry.src.exists():
            LOGGER.debug(f"Creating {entry.src}")
            entry.src.mkdir(parents=True, exist_ok=True)

    entries = [entry for entry in entries if entry.src]

    if os.environ.get("NAV_DIR"):
        host_nav_dir_path = pathlib.Path(os.environ.get("NAV_DIR"))
        LOGGER.warning(f"Using Model Navigator from {host_nav_dir_path}")
        container_nav_dir_path = pathlib.Path("/workspace/model_navigator")
        if host_nav_dir_path.exists():
            entries += [DirToMount(host_nav_dir_path, container_nav_dir_path, readonly=True)]
    return entries


def convert_model(
    *,
    config_path: pathlib.Path,
    workspace_path: pathlib.Path,
    model_name: str,
    model_path: pathlib.Path,
    parameters_to_override: typing.Optional[typing.Dict[str, typing.Any]] = None,
    verbose: bool = False,
):
    import sh
    from model_navigator.converter.utils import execute_sh_command

    init_job_env(verbose)

    parameters_to_override = parameters_to_override or {}
    parameters_suffix = "-".join([f"{key}_{value}" for key, value in parameters_to_override.items()])

    converted_model_path = workspace_path / f"{model_name}-{parameters_suffix}-converted.ft"
    converted_model_path = converted_model_path.resolve().absolute()
    if converted_model_path.exists():
        shutil.rmtree(converted_model_path)

    parameters = [[f"--{key.replace('_', '-')}", value] for key, value in parameters_to_override.items()]
    parameters_flattened = [item for pair in parameters for item in pair]

    model_navigator = sh.Command("model-navigator")
    convert_model_cmd = model_navigator.bake(
        "convert",
        "--config-path",
        config_path,
        "--model-path",
        model_path.resolve().absolute(),
        "--model-name",
        model_name,
        "--output-path",
        converted_model_path,
        "--override-workspace",
        "--verbose",
        *parameters_flattened,
    )

    cluster_suffix = get_cluster_suffix()
    log_path = workspace_path / "logs" / f"convert-{model_name}-{parameters_suffix}-{cluster_suffix}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w") as log_file:
            LOGGER.debug(f"Running {convert_model_cmd}")
            execute_sh_command(convert_model_cmd, log_file=log_file, verbose=True)
    except sh.ErrorReturnCode as e:
        msg = f"{e.stdout.decode('utf-8')}; more info in {log_path}"
        raise RuntimeError(msg)

    return converted_model_path


def convert_model_ft2ft(
    *,
    config_path: pathlib.Path,
    workspace_path: pathlib.Path,
    model_name: str,
    model_path: pathlib.Path,
    parameters_to_override: typing.Optional[typing.Dict[str, typing.Any]] = None,
    verbose: bool = False,
):
    from model_navigator.utils.config import YamlConfigFile

    init_job_env(verbose)

    parameters_to_override = parameters_to_override or {}
    parameters_suffix = "-".join([f"{key}_{value}" for key, value in parameters_to_override.items()])

    converted_model_path = workspace_path / f"{model_name}-{parameters_suffix}-converted.ft"
    converted_model_path = converted_model_path.resolve().absolute()
    if converted_model_path.exists():
        shutil.rmtree(converted_model_path)

    converted_model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_path, converted_model_path)
    ft_gpu_count = parameters_to_override.get("ft_gpu_counts", None)
    if ft_gpu_count is not None:
        import yaml

        with (converted_model_path / "meta.yaml").open("r") as meta_file:
            meta_info = yaml.load(meta_file, Loader=yaml.SafeLoader)
        meta_info["tensor_para_size"] = int(ft_gpu_count)
        with (converted_model_path / "meta.yaml").open("w") as meta_file:
            yaml.dump(meta_info, meta_file)

    config_path = converted_model_path.with_suffix(".nav.yaml")
    with YamlConfigFile(config_path) as config_file:
        config_file.save_key("launch_mode", "local")
        if ft_gpu_count is not None:
            config_file.save_key("ft_gpu_counts", [ft_gpu_count])

    return converted_model_path


def triton_config_model(
    *,
    config_path: pathlib.Path,
    workspace_path: pathlib.Path,
    model_path: pathlib.Path,
    variants: typing.List[Variant],
    verbose: bool = False,
):
    import sh
    from model_navigator.converter.utils import execute_sh_command

    init_job_env(verbose)
    triton_model_repository_paths = []
    for variant in variants:
        triton_model_repository_path = workspace_path / f"model_repo_{variant.extended_name}"
        if triton_model_repository_path.exists():
            for dir_entry in triton_model_repository_path.glob("*"):
                if dir_entry.is_dir():
                    shutil.rmtree(dir_entry)
                else:
                    dir_entry.unlink()

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

        model_navigator = sh.Command("model-navigator")
        triton_config_model_cmd = model_navigator.bake(
            "triton-config-model",
            "--config-path",
            config_path,
            "--model-path",
            model_path.resolve().absolute(),
            "--model-name",
            variant.extended_name,
            "--model-repository",
            triton_model_repository_path.resolve().absolute(),
            "--use-symlinks",
            "--verbose",
            *parameters,
        )

        cluster_suffix = get_cluster_suffix()
        log_path = workspace_path / "logs" / f"{cluster_suffix}_prepare_model_repository.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("w") as log_file:
                LOGGER.debug(f"Running {triton_config_model_cmd}")
                execute_sh_command(triton_config_model_cmd, log_file=log_file, verbose=True)
            triton_model_repository_paths.append(triton_model_repository_path)
        except sh.ErrorReturnCode as e:
            msg = f"{e.stdout.decode('utf-8')}; more info in {log_path}"
            raise RuntimeError(msg)

    return triton_model_repository_paths


def run_perf_test(
    *,
    config_path: pathlib.Path,
    workspace_path: pathlib.Path,
    bignlp_scripts_path: pathlib.Path,
    triton_endpoint_url: str,
    accuracy_tests: bool,
    performance_tests: bool,
    dataset_dir: pathlib.Path,
    variant: Variant,
    verbose: bool = False,
):
    import sh
    from model_navigator.converter.utils import execute_sh_command

    init_job_env(verbose)

    model_navigator = sh.Command("model-navigator")
    offline_latency_report_path = workspace_path / f"{variant.extended_name}-triton_performance_offline.csv"
    triton_offline_eval_model_cmd = model_navigator.bake(
        "triton-evaluate-model",
        "--config-path",
        config_path,
        "--server-url",
        triton_endpoint_url,
        "--batching-mode",
        "static",
        "--evaluation-mode",
        "online",
        "--model-name",
        variant.model_name,
        "--latency-report-file",
        offline_latency_report_path,
        verbose=verbose,
    )
    online_latency_report_path = workspace_path / f"{variant.extended_name}-triton_performance_online.csv"
    triton_online_eval_model_cmd = model_navigator.bake(
        "triton-evaluate-model",
        "--config-path",
        config_path,
        "--server-url",
        triton_endpoint_url,
        "--batching-mode",
        "dynamic",
        "--evaluation-mode",
        "online",
        "--model-name",
        variant.model_name,
        "--latency-report-file",
        online_latency_report_path,
        verbose=verbose,
    )

    accuracy_report_path = workspace_path / f"{variant.extended_name}-lambada_metrics.csv"
    accuracy_script_path = bignlp_scripts_path / "infer_scripts/evaluate_lambada.py"
    python3 = sh.Command("python3")
    accuracy_eval_cmd = python3.bake(
        accuracy_script_path,
        "-u",
        triton_endpoint_url.split("://")[1],
        "--protocol",
        "grpc",
        "--output_csv",
        accuracy_report_path,
        "-d",
        dataset_dir,
        "-b",
        4,
        "-m",
        variant.model_name,
        "--n-gram-disabled",
    )

    cluster_suffix = get_cluster_suffix()
    log_path = workspace_path / "logs" / f"{cluster_suffix}_benchmarking.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        reports_paths = []
        with log_path.open("w") as log_file:
            if performance_tests:
                LOGGER.debug(f"Running {triton_offline_eval_model_cmd}")
                execute_sh_command(triton_offline_eval_model_cmd, log_file=log_file, verbose=verbose)
                reports_paths.append(offline_latency_report_path)
                LOGGER.debug(f"Running {triton_online_eval_model_cmd}")
                execute_sh_command(triton_online_eval_model_cmd, log_file=log_file, verbose=verbose)
                reports_paths.append(online_latency_report_path)
            if accuracy_tests:
                LOGGER.debug(f"Running {accuracy_eval_cmd}")
                execute_sh_command(accuracy_eval_cmd, log_file=log_file, verbose=verbose)
                reports_paths.append(accuracy_report_path)
    except sh.ErrorReturnCode as e:
        msg = f"{e.stdout.decode('utf-8')}; more info in {log_path}"
        raise RuntimeError(msg)
    return "\n".join(f"\n{report_path}\n{report_path.read_text()}\n" for report_path in reports_paths)


def _patch_model_analyzer(paths):
    model_analyzer_dir = [
        list(pathlib.Path(sp).rglob("model_analyzer"))[0]
        for sp in site.getsitepackages()
        if list(pathlib.Path(sp).rglob("model_analyzer"))
    ][0]
    cmd = [
        "bash",
        "-c",
        f"cd {model_analyzer_dir} && patch -p2 < {paths.host_cwd / 'infer_scripts/inference_lib/patches/model_analyzer.patch'}",
    ]
    LOGGER.info(f"Patching model_analyzer: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True)
    LOGGER.info(f"returncode: {result.returncode}")
    LOGGER.info(f"stdout: \n{result.stdout.decode('utf-8')}")
    LOGGER.info(f"stderr: \n{result.stderr.decode('utf-8')}")


def run_profile(
    *,
    paths: Paths,
    workspace_path: pathlib.Path,
    config_path: pathlib.Path,
    model_repository_path: pathlib.Path,
    triton_endpoint_url: str,
    triton_metrics_url: str,
    verbose: bool = False,
):
    import sh
    from model_analyzer.triton.model.model_config import ModelConfig
    from model_navigator.converter.utils import execute_sh_command
    from model_navigator.model_analyzer.analyzer import AnalysisConfigGenerator
    from model_navigator.utils.workspace import DEFAULT_WORKSPACE_PATH, Workspace

    init_job_env(verbose)

    _patch_model_analyzer(paths)

    triton_config_path = list(model_repository_path.rglob("config.pbtxt"))[0]
    triton_config = ModelConfig.create_from_file(triton_config_path.parent.as_posix()).to_dict()

    input_len = int(triton_config["parameters"]["max_input_len"]["stringValue"])
    seq_len = int(triton_config["parameters"]["max_seq_len"]["stringValue"])
    output_len = seq_len - input_len
    vocab_size = int(triton_config["parameters"]["end_id"]["stringValue"])

    max_shapes = [f"INPUT_ID=-1,1,{input_len}", "REQUEST_INPUT_LEN=-1,1", "REQUEST_OUTPUT_LEN=-1,1"]
    value_ranges = [
        f"INPUT_ID=1,{vocab_size}",
        f"REQUEST_INPUT_LEN={input_len},{input_len}",
        f"REQUEST_OUTPUT_LEN={output_len},{output_len}",
    ]
    dtypes = ["INPUT_ID=uint32", "REQUEST_INPUT_LEN=uint32", "REQUEST_OUTPUT_LEN=uint32"]
    model_navigator = sh.Command("model-navigator")
    profile_cmd = model_navigator.bake(
        "profile",
        "--config-path",
        config_path,
        "--triton-endpoint-url",
        triton_endpoint_url,
        "--triton-metrics-url",
        triton_metrics_url,
        "--model-repository",
        model_repository_path,
        "--max-shapes",
        *max_shapes,
        "--value-ranges",
        *value_ranges,
        "--dtypes",
        *dtypes,
        "--verbose",
    )

    cluster_suffix = get_cluster_suffix()
    log_path = workspace_path / "logs" / f"profile_{cluster_suffix}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w") as log_file:
            LOGGER.debug(f"Running {profile_cmd}")
            execute_sh_command(profile_cmd, log_file=log_file, verbose=verbose)

        # WAR: MA doesn't create model-store
        nav_workspace = Workspace(workspace_path / DEFAULT_WORKSPACE_PATH)
        analyzer_model_repo_path = (
            AnalysisConfigGenerator(workspace=nav_workspace, analysis_config=None).analyzer_path / "interim-model-store"
        )
        analyzer_model_repo_path.mkdir(parents=True, exist_ok=True)

        # remaps container directory to host directory
        model_repo_dir = paths.host_triton_model_repository_path / triton_config_path.parent.name
        (analyzer_model_repo_path / model_repo_dir.name).symlink_to(model_repo_dir)

    except sh.ErrorReturnCode as e:
        msg = f"{e.stdout.decode('utf-8')}; more info in {log_path}"
        raise RuntimeError(msg)
    return None


def run_analyze(
    *,
    paths: Paths,
    workspace_path: pathlib.Path,
    top_n_configs: int,
    max_latency_ms: int,
    verbose: bool = False,
):
    import sh
    from model_navigator.converter.utils import execute_sh_command
    from model_navigator.model_analyzer.analyzer import AnalysisConfigGenerator
    from model_navigator.utils.workspace import DEFAULT_WORKSPACE_PATH, Workspace

    init_job_env(verbose)

    _patch_model_analyzer(paths)

    nav_workspace = Workspace(workspace_path / DEFAULT_WORKSPACE_PATH)
    interim_analyzer_model_repo_path = (
        AnalysisConfigGenerator(workspace=nav_workspace, analysis_config=None).analyzer_path / "interim-model-store"
    )

    analyzer_model_repo_path = AnalysisConfigGenerator(
        workspace=nav_workspace, analysis_config=None
    ).output_model_repository_path

    if interim_analyzer_model_repo_path.exists():
        if analyzer_model_repo_path.exists():
            shutil.rmtree(analyzer_model_repo_path)
        interim_analyzer_model_repo_path.rename(analyzer_model_repo_path)

    inference_output_fields = [
        "model_name",
        "batch_size",
        "concurrency",
        "model_config_path",
        "backend_parameters",
        "dynamic_batch_sizes",
        "satisfies_constraints",
        "perf_throughput",
        "perf_latency_p50",
        "perf_latency_p95",
        "perf_latency_p99",
    ]

    model_navigator = sh.Command("model-navigator")
    analyzer_cmd = model_navigator.bake(
        "analyze",
        "--model-repository",
        analyzer_model_repo_path,
        "--top-n-configs",
        top_n_configs,
        "--max-latency-ms",
        max_latency_ms,
        "--inference-output-fields",
        *inference_output_fields,
    )

    cluster_suffix = get_cluster_suffix()
    log_path = workspace_path / "logs" / f"analyze_{cluster_suffix}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w") as log_file:
            LOGGER.debug(f"Running {analyzer_cmd}")
            execute_sh_command(analyzer_cmd, log_file=log_file, verbose=verbose)
    except sh.ErrorReturnCode as e:
        msg = f"{e.stdout.decode('utf-8')}; more info in {log_path}"
        raise RuntimeError(msg)

    summary = log_path.read_text()
    return summary
