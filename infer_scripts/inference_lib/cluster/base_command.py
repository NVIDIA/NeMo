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
import enum
import ipaddress
import json
import logging
import os
import pathlib
import shlex
import shutil
import subprocess
import tempfile
import textwrap
import time
import typing

from .executor import BaseExecutor, BaseInfoCollector, BaseSubmissionGenerator
from .job import BaseJob, DirToMount, JobDefinition, JobInfo, JobStatus, MountMode
from .utils import execute_command_and_get_json_result, execute_command_and_get_result

LOGGER = logging.getLogger(__name__)

_STATUS_MAPPING = {
    "CREATED": JobStatus.QUEUED,
    "QUEUED": JobStatus.QUEUED,
    "STARTING": JobStatus.STARTING,
    "RUNNING": JobStatus.RUNNING,
    "FAILED": JobStatus.FAILED,
    "FAILED_RUN_LIMIT_EXCEEDED": JobStatus.FAILED,
    "FINISHED_SUCCESS": JobStatus.FINISHED_SUCCESS,
    "KILLED_BY_SYSTEM": JobStatus.KILLED_BY_SYSTEM,
    "KILLED_BY_USER": JobStatus.KILLED_BY_USER,
    "PREEMPTED": JobStatus.PREEMPTED,
    "PENDING_TERMINATION": JobStatus.UNKNOWN,
    "TASK_LOST": JobStatus.UNKNOWN,
    "UNKNOWN": JobStatus.UNKNOWN,
}

_MAX_JOB_NAME_LEN = 128


class BaseCommandFormatType(enum.Enum):
    ASCII = "ascii"
    CSV = "csv"
    JSON = "json"


class BaseCommandArrayType(enum.Enum):
    MPI = "MPI"
    PARALLEL = "PARALLEL"
    PYTORCH = "PYTORCH"
    HOROVOD = "HOROVOD"


@dataclasses.dataclass
class BaseCommandJobSpec:
    commandline: str
    description: str
    image: str
    instance: str
    name: str
    result: pathlib.Path
    ports: typing.Optional[typing.List[int]] = None
    workspaces: typing.Optional[typing.List[DirToMount]] = None
    ace: typing.Optional[str] = None
    array_type: typing.Optional[BaseCommandArrayType] = None
    coscheduling: typing.Optional[bool] = None
    datasetsids: typing.Optional[typing.List[DirToMount]] = None
    entrypoint: typing.Optional[str] = None
    format_type: typing.Optional[BaseCommandFormatType] = None
    replicas: typing.Optional[int] = None
    start_deadline: typing.Optional[str] = None
    total_runtime: typing.Optional[str] = None

    @classmethod
    def from_job_definition(cls, executor, job_definition: JobDefinition) -> "BaseCommandJobSpec":
        # here instance, start_deadline
        parameters_from_executor = executor.get_cluster_parameters_for_job_definition(job_definition)

        def _get_mount_from_job_definition(requested_mount_):
            # check if possible to map local path to remote
            remote_path = executor.map_local_to_remote_path(requested_mount_)
            workspace_name_or_id = remote_path.split(":")[0]
            mounted_storage = {mount.src: mount for mount in executor.mounted_storages}
            return mounted_storage[workspace_name_or_id]

        def _format_total_runtime(max_time_s: int) -> str:
            return f"{max_time_s}s"

        workspaces_to_mount = list(set(map(_get_mount_from_job_definition, job_definition.directories_to_mount)))

        assert isinstance(job_definition.commands, list), "Currently only list of strings is supported"
        command = f"bash -c {shlex.quote(' && '.join(job_definition.commands))}"

        mpirun_prefix = (
            r"mpirun --allow-run-as-root -np ${NGC_ARRAY_SIZE:-1} --map-by ppr:"
            + str(job_definition.tasks_number_per_node or 1)
            + ":node "
        )

        separate_commands = (
            (job_definition.setup_commands or []) + [mpirun_prefix + command] + (job_definition.clean_commands or [])
        )
        commandline = "\n".join(separate_commands)

        return BaseCommandJobSpec(
            **parameters_from_executor,
            commandline=commandline,
            description=job_definition.description,
            image=job_definition.container_image,
            name=job_definition.name[:_MAX_JOB_NAME_LEN],
            result=pathlib.Path("/result"),
            ports=job_definition.ports,
            array_type=BaseCommandArrayType.MPI,
            workspaces=workspaces_to_mount,
            datasetsids=None,
            format_type=BaseCommandFormatType.JSON,
            replicas=job_definition.tasks_number,
            total_runtime=_format_total_runtime(job_definition.max_time_s),
        )


class BaseCommandJob(BaseJob):
    def __init__(self, job_id: str, executor):
        super().__init__(job_id, executor)
        self._ip_address = None

    def get_endpoint_url(self, scheme: str, port: int) -> str:
        if not self._ip_address:
            result_content = self.exec(
                'python3 -c "import socket; '
                "s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); "
                "s.connect(('8.8.8.8', 80)); "
                'print(s.getsockname()[0])"',
            )

            ip_address = None
            for line in result_content.splitlines():
                try:
                    ip_address = ipaddress.ip_address(line.strip())
                    break
                except ValueError:
                    pass

            if not ip_address:
                raise RuntimeError(
                    f"Could not obtain ip address from job container; output from container: {result_content}"
                )
            self._ip_address = ip_address

        return f"{scheme}://{self._ip_address}:{port}"

    def sync_logs(self):
        LOGGER.info(f"[{self.job_id}] Waiting for logs and downloading them into {self.log_path}")
        with tempfile.TemporaryDirectory() as temp_dir:
            cwd = pathlib.Path.cwd()
            os.chdir(temp_dir)
            try:
                while not self.log_path.exists():
                    log_filename = "joblog.log"

                    try:
                        execute_command_and_get_result(
                            f"ngc result download --file /{log_filename} {self._job_id}",
                            timeout_s=self._executor._command_timeout_s,
                        )
                    except Exception:
                        pass

                    tmp_log_path = pathlib.Path(temp_dir) / f"{self.job_id}/{log_filename}"
                    if tmp_log_path.exists():
                        shutil.copy(tmp_log_path, self.log_path)
                    else:
                        time.sleep(5)

            except Exception as e:
                LOGGER.warning(e)
            finally:
                os.chdir(cwd)


class BaseCommandSubmissionGenerator(BaseSubmissionGenerator):

    _PARAMETERS_NAMES_MAP = {"ports": "port", "workspaces": "workspace"}
    _PARAMETERS_NAMES_WITHOUT_HYPHEN_CHANGE_LIST = ["format_type"]

    file_suffix = ".sh"

    def get_submission_file_content(self, job_spec: BaseCommandJobSpec) -> str:
        sample_submission_command = self.get_submission_command(job_spec, pathlib.Path("<path_to_this_file>"))
        return "\n".join(
            [
                "#!/usr/bin/env bash",
                *textwrap.indent(sample_submission_command, "# ").splitlines(),
                "set -e",
                "set -x",
                job_spec.commandline,
            ]
        )

    def get_submission_command(self, job_spec: BaseCommandJobSpec, submission_file_path: pathlib.Path):
        def _rewrite_parameter(name_, value_) -> typing.List[typing.List[str]]:

            name_ = self._PARAMETERS_NAMES_MAP.get(name_, name_)
            if name_ not in self._PARAMETERS_NAMES_WITHOUT_HYPHEN_CHANGE_LIST:
                name_ = name_.replace("_", "-")

            def _map_value(v):
                if isinstance(v, pathlib.Path):
                    v = v.as_posix()
                elif isinstance(v, enum.Enum):
                    v = v.value
                return shlex.quote(str(v))

            if value_ is None:
                cli_args_pairs = []
            else:
                # handle lists as multiple parameters
                if not isinstance(value_, list):
                    value_ = [value_]
                cli_args_pairs = [[name_] if isinstance(item, bool) else [name_, _map_value(item)] for item in value_]

            return cli_args_pairs

        # set job id env as job id in submission filename
        submission_file_with_env_path = self._executor.paths.get_submission_file_path(
            job_id=f"${{{self._executor._env_variable_with_job_id}}}", suffix=submission_file_path.suffix
        )

        commandline = (
            f'timeout 60 bash -c "while [ ! -f "{submission_file_with_env_path}" ]; do sleep 1; done" && '
            f"stdbuf -oL bash {submission_file_with_env_path} || "
            f'{{ stdbuf -oL echo "Could not found {submission_file_with_env_path} file."; exit -1; }}'
        )

        parameters = dict((field.name, getattr(job_spec, field.name)) for field in dataclasses.fields(job_spec))
        parameters["commandline"] = commandline
        cli_args = [
            "--" + " ".join(cli_arg_pairs)
            for name, value in parameters.items()
            for cli_arg_pairs in _rewrite_parameter(name, value)
        ]
        parameters_cli = " \\\n".join(cli_args)
        parameters_cli = textwrap.indent(parameters_cli, "    ")

        return "ngc batch run \\\n" + parameters_cli

    def parse_job_id_from_submission_command_result(self, result: str) -> str:
        result = json.loads(result)
        return str(result["id"])


class BaseCommandInfoCollector(BaseInfoCollector):
    _get_info_command = "ngc batch list --format_type=json"

    def parse_get_info_command_result(self, info_command_result: str) -> typing.List[JobInfo]:
        info_command_result = json.loads(info_command_result)
        return [
            JobInfo(
                job_id=str(entry["id"]),
                name=entry["jobDefinition"]["name"],
                description=entry["jobDefinition"].get("description", ""),
                state=_STATUS_MAPPING[entry["jobStatus"].get("status")],
            )
            for entry in info_command_result
        ]


class BaseCommandExecutor(BaseExecutor):
    _job_class = BaseCommandJob
    _job_spec_class = BaseCommandJobSpec
    _info_collector_class = BaseCommandInfoCollector
    _submission_generator_class = BaseCommandSubmissionGenerator

    _env_variable_with_job_id = "NGC_JOB_ID"
    _command_timeout_s = 180
    _cancel_command = "ngc batch kill {job_id}"
    _exec_command = "ngc batch exec --commandline {command} {job_id}"

    def __init__(self, cluster_dir_path: pathlib.Path, cluster_config):
        super().__init__(cluster_dir_path, cluster_config)
        self._available_workspaces = None
        self._mounted_storages = None
        self._get_cluster_details()

    def get_cluster_parameters_for_job_definition(self, job_definition: JobDefinition) -> typing.Dict[str, typing.Any]:
        instance = (
            self._cluster_config["instance_with_gpu"]
            if job_definition.gpus_number_per_task
            else self._cluster_config["instance_without_gpu"]
        )
        parameters = {"instance": instance}
        LOGGER.debug(f"BaseCommandJobSpec parameters from executor {self.name()}: {parameters}")
        return parameters

    def _get_cluster_details(self):
        self._available_workspaces = self._get_available_workspaces()
        # having available workspaces - check if any of them are mounted
        self._mounted_storages = self._get_mounted_storages()
        LOGGER.debug(f"Mounted storages: {', '.join(str(mount) for mount in self._mounted_storages)}")

    def _get_available_workspaces(self):
        workspace_list_result = execute_command_and_get_json_result(
            f"ngc workspace list --format_type json", timeout_s=self._command_timeout_s
        )
        return [item["name"] for item in workspace_list_result if "name" in item] + [
            item["id"] for item in workspace_list_result if "id" in item
        ]

    def _get_mounted_storages(self):
        assert self._available_workspaces is not None

        mount_result = execute_command_and_get_result("mount", timeout_s=self._command_timeout_s)
        mounted_storage = mount_result.splitlines()
        mounted_storage = [
            item
            for item in mounted_storage
            if "sshfs" in item
            and any(workspace_name_or_id in item for workspace_name_or_id in self._available_workspaces)
        ]

        def _extract_name_and_local_path(entry):
            # ex 4HmhA6p0S-K0MKVj5Lkpcw on /foo/b type fuse.sshfs (rw,nosuid,nodev,relatime,user_id=1000,group_id=1000)
            items = entry.split(" ")
            is_writable = "rw" in items[5]
            mode = MountMode.RW if is_writable else MountMode.RO
            return DirToMount(src=items[0], dst=pathlib.Path(items[2]), mode=mode)

        return [_extract_name_and_local_path(entry) for entry in mounted_storage]

    @property
    def mounted_storages(self):
        assert self._mounted_storages is not None
        return self._mounted_storages

    def map_local_to_remote_path(self, local_path: pathlib.Path) -> str:
        assert self._mounted_storages is not None

        def _is_relative_to(path, to):
            try:
                return bool(path.relative_to(to))
            except ValueError:
                return False

        matching_storage = {mount for mount in self._mounted_storages if _is_relative_to(local_path, mount.dst)}
        if not matching_storage:
            error_message = f"Could not get remote path as {local_path} is not internal of any storage."
            if self._mounted_storages:
                mounted_storage_text = ", ".join(map(str, self._mounted_storages))
                error_message += f" Mounted storages: {mounted_storage_text}."

            raise ValueError(error_message)
        elif len(matching_storage) > 1:
            matching_storage_text = ", ".join(map(str, matching_storage))
            LOGGER.warning(f"More than one matching storage found: {matching_storage_text}. Use the first one.")

        mount = list(matching_storage)[0]
        return f"{mount.src}:{local_path.relative_to(mount.dst)}"
