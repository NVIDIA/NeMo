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
import datetime
import logging
import pathlib
import re
import shlex
import typing

from .executor import BaseExecutor, BaseInfoCollector, BaseSubmissionGenerator, ClusterNavigatorException
from .job import BaseJob, JobDefinition, JobInfo, JobStatus
from .utils import execute_command_and_get_result

LOGGER = logging.getLogger(__name__)

_STATE_MAPPING = {
    "PENDING": JobStatus.QUEUED,
    "RUNNING": JobStatus.RUNNING,
    "FAILED": JobStatus.FAILED,
    "BOOT_FAIL": JobStatus.FAILED,
    "NODE_FAIL": JobStatus.FAILED,
    "OUT_OF_MEMORY": JobStatus.FAILED,
    "COMPLETED": JobStatus.FINISHED_SUCCESS,
    "CANCELLED": JobStatus.KILLED_BY_USER,
    "TIMEOUT": JobStatus.KILLED_BY_SYSTEM,
    "PREEMPTED": JobStatus.PREEMPTED,
    "UNKNOWN": JobStatus.UNKNOWN,
}


def _rewrite_docker_image(docker_image):
    if "#" not in docker_image:
        docker_image = "#".join(docker_image.split("/", maxsplit=1))
    return docker_image


def _format_container_mounts(mounts: typing.List[pathlib.Path]) -> str:
    mounts = sorted(set(mounts))
    for mount in mounts:
        mount.mkdir(parents=True, exist_ok=True)
    return ",".join([f"{mount}:{mount}" for mount in mounts])


def _format_slurm_time(time_delta: datetime.timedelta):
    mm, ss = divmod(time_delta.seconds, 60)
    hh, mm = divmod(mm, 60)
    return f"{time_delta.days}-{hh:02d}:{mm:02d}:{ss:02d}"


def _expand_id_suffix(suffix_parts: str) -> typing.List[str]:
    """
    (import from submitit.slurm.slurm package)
    Parse the a suffix formatted like "1-3,5,8" into
    the list of numeric values 1,2,3,5,8.
    """
    suffixes = []
    for suffix_part in suffix_parts.split(","):
        if "-" in suffix_part:
            low, high = suffix_part.split("-")
            int_length = max(len(low), len(high))
            suffixes.extend([f"{num:0{int_length}}" for num in range(int(low), int(high) + 1)])
        else:
            suffixes.append(suffix_part)
    return suffixes


def _parse_node_group(node_list: str, pos: int, parsed: typing.List[str]) -> int:
    """
    (import from submitit.slurm.slurm package)
    Parse a node group of the form PREFIX[1-3,5,8] and return
    the position in the string at which the parsing stopped
    """
    prefixes = [""]
    while pos < len(node_list):
        c = node_list[pos]
        if c == ",":
            parsed.extend(prefixes)
            return pos + 1
        if c == "[":
            last_pos = node_list.index("]", pos)
            suffixes = _expand_id_suffix(node_list[pos + 1 : last_pos])
            prefixes = [prefix + suffix for prefix in prefixes for suffix in suffixes]
            pos = last_pos + 1
        else:
            for i, prefix in enumerate(prefixes):
                prefixes[i] = prefix + c
            pos += 1
    parsed.extend(prefixes)
    return pos


def parse_node_list(node_list: str) -> typing.List[str]:
    # (import from submitit.slurm.slurm package)
    try:
        pos = 0
        parsed = []
        while pos < len(node_list):
            pos = _parse_node_group(node_list, pos, parsed)
        return parsed
    except ValueError as e:
        raise ValueError(f"Unrecognized format for node list: {node_list}", e) from e


def _parse_sacct_result(sacct_result: str) -> typing.List[typing.Dict[str, str]]:
    header, *rows = sacct_result.splitlines()
    if not rows:
        return []

    sep = "|"
    names = header.split(sep)

    def _parse_entry(entry) -> typing.Optional[typing.Dict[str, str]]:
        items = entry.split(sep)
        job_id = items[names.index("JobID")]
        parsed_entry = None
        if job_id and "." not in job_id:
            if "State" in names:
                # to correctly parse also CANCELLED by <job_id>
                original_state = items[names.index("State")]
                items[names.index("State")] = original_state.split(" ")[0]
            parsed_entry = dict(zip(names, items))
        return parsed_entry

    return [parsed_entry for parsed_entry in map(_parse_entry, rows) if parsed_entry]


class PyxisInfoCollector(BaseInfoCollector):
    _get_info_command = "sacct -o JobID,JobName,Comment,State --parsable2"

    def parse_get_info_command_result(self, info_command_result: str) -> typing.List[JobInfo]:
        entries = _parse_sacct_result(info_command_result)
        return [
            JobInfo(
                job_id=entry["JobID"],
                name=entry["JobName"],
                description=entry["Comment"],
                state=_STATE_MAPPING[entry["State"]],
            )
            for entry in entries
        ]


class PyxisJob(BaseJob):
    def __init__(self, job_id: str, executor):
        super().__init__(job_id, executor)
        self._first_node_name = None

    def get_endpoint_url(self, scheme: str, port: int) -> str:
        if not self._first_node_name:
            sacct_command = f"sacct -o JobID,NodeList --parsable2 -j {self.job_id}"
            sacct_result = execute_command_and_get_result(sacct_command, timeout_s=self._executor._command_timeout_s)
            entries = _parse_sacct_result(sacct_result)
            entries = {entry["JobID"]: entry["NodeList"] for entry in entries}
            job_nodes_list = entries[self.job_id]
            nodes = parse_node_list(job_nodes_list)
            self._first_node_name = nodes[0]
        return f"{scheme}://{self._first_node_name}:{port}"


@dataclasses.dataclass
class SlurmClusterParameters:
    account: typing.Optional[str] = None
    partition: typing.Optional[str] = None
    srun_args: typing.Optional[str] = None
    support_gpus_allocation: typing.Optional[bool] = True


@dataclasses.dataclass
class PyxisJobSpec:
    job_name: str
    comment: str
    commandline: str
    container_image: str
    partition: str
    time: str
    output: pathlib.Path
    account: typing.Optional[str] = None
    container_mounts: typing.Optional[str] = None
    container_workdir: typing.Optional[pathlib.Path] = None
    srun_args: typing.Optional[str] = None
    nodes: int = 1
    ntasks_per_node: typing.Optional[int] = None
    cpus_per_task: typing.Optional[int] = None
    gpus_per_task: typing.Optional[int] = None
    mem_per_cpu: typing.Optional[str] = None
    mem_per_gpu: typing.Optional[str] = None
    wckey: typing.Optional[str] = None
    open_mode: str = "append"
    signal: str = f"USR1@90"

    @classmethod
    def from_job_definition(cls, executor, job_definition: JobDefinition):
        cluster_parameters: SlurmClusterParameters = executor.get_cluster_parameters_for_job_definition(job_definition)

        gpus_per_task = job_definition.gpus_number_per_task if cluster_parameters.support_gpus_allocation else None
        if not job_definition.cpus_number_per_task and not gpus_per_task and job_definition.mem_number_gb_per_task:
            LOGGER.warning(
                f"Ignore JobDefinition.mem_number_gb_per_task={job_definition.mem_number_gb_per_task} "
                f"because of missing cpus_number_per_task and gpus_number_per_task JobDefinition parameters"
            )

        mem_per_cpu = None
        mem_per_gpu = None
        if job_definition.mem_number_gb_per_task:
            if job_definition.cpus_number_per_task:
                mem_per_cpu = f"{job_definition.mem_number_gb_per_task / job_definition.cpus_number_per_task:0.1f}G"
            if gpus_per_task:
                mem_per_gpu = f"{job_definition.mem_number_gb_per_task / gpus_per_task :0.1f}G"

        def _extract_path(p: typing.Optional[pathlib.Path] = None):
            if p:
                return p.absolute().resolve()
            else:
                return None

        assert isinstance(job_definition.commands, list), "Currently only list of strings is supported"
        command = f"bash -c {shlex.quote(' && '.join(job_definition.commands))}"

        separate_commands = (job_definition.setup_commands or []) + [command] + (job_definition.clean_commands or [])
        commandline = "\n".join(separate_commands)

        return cls(
            account=cluster_parameters.account,
            partition=cluster_parameters.partition,
            srun_args=cluster_parameters.srun_args,
            job_name=shlex.quote(job_definition.name),
            comment=shlex.quote(job_definition.description),
            container_image=_rewrite_docker_image(job_definition.container_image),
            container_workdir=_extract_path(job_definition.workdir_path),
            container_mounts=_format_container_mounts(job_definition.directories_to_mount),
            output=executor.paths.get_log_path("%j"),
            commandline=commandline,
            time=_format_slurm_time(datetime.timedelta(seconds=job_definition.max_time_s)),
            nodes=job_definition.tasks_number // (job_definition.tasks_number_per_node or 1),
            ntasks_per_node=job_definition.tasks_number_per_node,
            cpus_per_task=job_definition.cpus_number_per_task,
            gpus_per_task=gpus_per_task,
            mem_per_cpu=mem_per_cpu,
            mem_per_gpu=mem_per_gpu,
        )


class PyxisJobSubmissionGenerator(BaseSubmissionGenerator):
    _NOT_SBATCH_PARAMETERS = ["commandline", "container_image", "container_mounts", "container_workdir", "srun_args"]
    file_suffix = ".sh"

    def get_submission_file_content(self, job_spec: PyxisJobSpec) -> str:
        parameters = {
            field.name: getattr(job_spec, field.name)
            for field in dataclasses.fields(job_spec)
            if getattr(job_spec, field.name) is not None and field.name not in self._NOT_SBATCH_PARAMETERS
        }
        lines = ["#!/usr/bin/env bash", "", "# parameters"]
        lines += [
            f"#SBATCH --{name.replace('_', '-')}"
            f"{'' if isinstance(parameters[name], bool) and parameters[name] else f'={parameters[name]}'}"
            for name in sorted(parameters)
        ]

        srun_full_command_command = (
            f"srun "
            f"{job_spec.srun_args or ''} \\\n"
            f"  --output {job_spec.output} \\\n"
            f"  --container-image {job_spec.container_image} \\\n"
        )
        if job_spec.container_mounts:
            srun_full_command_command += f"  --container-mounts {job_spec.container_mounts} \\\n"
        if job_spec.container_workdir:
            srun_full_command_command += [f"  --container-workdir {job_spec.container_workdir} \\\n"]
        srun_full_command_command += f"  --unbuffered \\\n  {job_spec.commandline}"

        lines += ["", "# command", srun_full_command_command]
        return "\n".join(lines)

    def get_submission_command(self, job_spec: PyxisJobSpec, submission_file_path: pathlib.Path) -> str:
        return f"sbatch {submission_file_path}"

    def parse_job_id_from_submission_command_result(self, result: str) -> str:
        output = re.search(r"job (?P<id>[0-9]+)", result)
        if output is None:
            raise ClusterNavigatorException(f"Could not make sense of sbatch output '{result}'")
        return output.group("id")


class PyxisExecutor(BaseExecutor):
    _job_class = PyxisJob
    _job_spec_class = PyxisJobSpec
    _submission_generator_class = PyxisJobSubmissionGenerator
    _info_collector_class = PyxisInfoCollector
    _env_variable_with_job_id = "SLURM_JOB_ID"
    _info_refresh_interval_s = 5
    _cancel_command = "scancel {job_id}"

    def get_cluster_parameters_for_job_definition(self, job_definition: JobDefinition) -> SlurmClusterParameters:
        return SlurmClusterParameters(
            account=self._cluster_config.get("account"),
            partition=self._cluster_config.get("partition"),
            srun_args=" ".join(self._cluster_config.get("srun_args", [])),
            support_gpus_allocation=self._cluster_config.get(
                "support_gpus_allocation", self._cluster_config.get("enable_gpus_allocation", True)
            ),
        )
