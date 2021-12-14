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
import abc
import atexit
import copy
import logging
import pathlib
import shlex
import subprocess
import textwrap
import threading
import time
import typing
from typing import Any

from .job import BaseJob, JobDefinition, JobInfo
from .utils import ClusterNavigatorPathsGenerator, execute_command_and_get_result

LOGGER = logging.getLogger(__name__)


class ClusterNavigatorException(Exception):
    def __init__(self, message: str, failed_jobs: typing.Optional[typing.List[BaseJob]] = None):
        self._message = message
        self._failed_jobs = failed_jobs

    def __str__(self):
        message_lines = [self._message]
        failed_jobs = self._failed_jobs or []
        for failed_job in failed_jobs:
            if failed_job.log_path.exists():
                NUM_RECENT_LOGS_IN_ERROR = 64

                logs_lines = failed_job.log_path.read_text().splitlines()
                logs_content = textwrap.indent("\n".join(logs_lines[-NUM_RECENT_LOGS_IN_ERROR:]), "    ")
                logs_message = f"Recent logs:\n\n{logs_content}\n\nFull logs may be found in {failed_job.log_path}\n"
            else:
                logs_message = "Use xxxx command to obtain logs."

            message_lines.append(
                f"  - [{failed_job.job_id}] "
                f"failed {failed_job.info.description or failed_job.info.name} job. "
                f"{logs_message}"
            )

        return "\n".join(message_lines)

    @property
    def message(self):
        return self._message

    @property
    def failed_jobs(self) -> typing.Optional[typing.List[BaseJob]]:
        return self._failed_jobs


class BaseInfoCollector:
    _get_info_command = None

    def __init__(self, refresh_interval_s: int, command_timeout_s: int):
        self._info_map = {}
        self._info_map_condition = threading.Condition()
        self._command_timeout_s = command_timeout_s
        self._refresh_interval_s = refresh_interval_s
        self._last_update = float("-inf")

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def force_update(self):
        with self._info_map_condition:
            current_update = self._last_update
            self._info_map_condition.notify_all()
            result = self._info_map_condition.wait_for(
                predicate=lambda: self._last_update > current_update, timeout=self._refresh_interval_s
            )
            if not result:
                LOGGER.warning(f"Could not jobs infos update (timeout_s={self._refresh_interval_s})")

    def get_info(self, job_id: str) -> typing.Optional[JobInfo]:
        assert isinstance(job_id, str)

        with self._info_map_condition:
            if job_id not in self._info_map:
                LOGGER.debug(f"Missing info for {job_id} - forcing load from cluster")
                self.force_update()
            return self._info_map.get(job_id)

    def _run(self):
        while True:
            batch_list_result = None
            started_update = time.time()
            try:
                batch_list_result = execute_command_and_get_result(
                    self._get_info_command, timeout_s=self._command_timeout_s
                )
            except subprocess.TimeoutExpired:
                LOGGER.warning(f"Could not obtain jobs list in {self._command_timeout_s}s")
            except ClusterNavigatorException as e:
                LOGGER.warning(f"Could not obtain jobs list due to {e}")

            with self._info_map_condition:
                if batch_list_result:
                    infos = self.parse_get_info_command_result(batch_list_result)
                    for info in infos:
                        assert isinstance(info.job_id, str)
                        self._info_map[info.job_id] = info
                    self._last_update = started_update
                    self._info_map_condition.notify_all()
                timeout_s = max(0, int(self._refresh_interval_s - (time.time() - started_update)))
                if timeout_s > 0:
                    self._info_map_condition.wait(timeout=timeout_s)

    @abc.abstractmethod
    def parse_get_info_command_result(self, info_command_result: str) -> typing.List[JobInfo]:
        pass


class BaseSubmissionGenerator(abc.ABC):
    def __init__(self, executor: "BaseExecutor"):
        self._executor = executor

    @property
    @abc.abstractmethod
    def file_suffix(self):
        pass

    @abc.abstractmethod
    def get_submission_file_content(self, job_spec) -> str:
        pass

    @abc.abstractmethod
    def get_submission_command(self, job_spec, submission_file_path: pathlib.Path) -> str:
        pass

    @abc.abstractmethod
    def parse_job_id_from_submission_command_result(self, result: str) -> str:
        pass


class ClusterExecutor:
    def __new__(cls, cluster_dir_path: pathlib.Path, cluster_config) -> Any:
        from .base_command import BaseCommandExecutor
        from .pyxis import PyxisExecutor

        EXECUTOR_MAP = {
            "pyxis": PyxisExecutor,
            "slurm": PyxisExecutor,
            "base_command": BaseCommandExecutor,
        }

        cluster_type = cluster_config.get("type")
        if not cluster_type:
            raise ValueError(f"Cluster config should have 'type' key")

        try:
            ExecutorCls = EXECUTOR_MAP[cluster_type]
        except KeyError:
            raise ValueError(
                f"Unknown {cluster_type} cluster type. Available types: {', '.join(name for name in EXECUTOR_MAP)}"
            )

        return ExecutorCls(cluster_dir_path=cluster_dir_path, cluster_config=cluster_config)


class BaseExecutor(abc.ABC):
    _job_class = None
    _job_spec_class = None
    _submission_generator_class = None
    _info_collector_class = None
    _paths_generator_class = ClusterNavigatorPathsGenerator

    # environment variables
    _env_variable_with_job_id = None

    # timeouts
    _command_timeout_s = 60
    _info_refresh_interval_s = 30

    # command patterns
    _cancel_command = None
    _exec_command = None

    def __init__(self, cluster_dir_path: pathlib.Path, cluster_config):
        self._cluster_config = cluster_config
        self.paths = self._paths_generator_class(cluster_dir_path)
        self._info_collector = self._info_collector_class(
            refresh_interval_s=self._info_refresh_interval_s, command_timeout_s=self._command_timeout_s
        )
        cluster_dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def name(cls) -> str:
        n = cls.__name__
        if n.endswith("Executor"):
            n = n.rstrip("Executor")
        return n.lower()

    def get_job(self, job_id) -> BaseJob:
        return self._job_class(job_id, executor=self)

    def get_info(self, job_id: str, force_update: bool = False):
        if force_update:
            self._info_collector.force_update()
        return self._info_collector.get_info(job_id)

    @abc.abstractmethod
    def get_cluster_parameters_for_job_definition(self, job_definition: JobDefinition) -> typing.Dict[str, typing.Any]:
        pass

    def submit(self, job_definition: JobDefinition):
        # generate submission file
        job_definition = copy.copy(job_definition)
        job_definition.directories_to_mount += [self.paths.cluster_dir]  # that job have access to submission files
        job_spec = self._job_spec_class.from_job_definition(self, job_definition)

        submission_generator: BaseSubmissionGenerator = self._submission_generator_class(self)
        submission_file_content = submission_generator.get_submission_file_content(job_spec)
        submission_file_tmp_path = self.paths.get_temporary_file_path(suffix=submission_generator.file_suffix)
        submission_file_tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with submission_file_tmp_path.open("w") as submission_file:
            submission_file.write(submission_file_content)

        # submit job
        submission_command = submission_generator.get_submission_command(job_spec, submission_file_tmp_path)
        submission_result = execute_command_and_get_result(
            submission_command, check=True, timeout_s=self._command_timeout_s
        )
        job_id = submission_generator.parse_job_id_from_submission_command_result(submission_result)
        assert isinstance(job_id, str)
        job = self.get_job(job_id)

        # cleanup tasks
        atexit.register(self.cancel, job_id=job_id, check=False)
        self._info_collector.force_update()
        self.paths.replace_job_id_and_rename_file(submission_file_tmp_path, job_id=job_id)

        return job

    def run(self, job_definition: JobDefinition, dependencies: typing.Optional[typing.List[BaseJob]] = None):
        job = self.submit(job_definition)
        job.wait(dependencies)

        dependencies = dependencies or []

        for job_for_log_sync in [job] + dependencies:
            job_for_log_sync.sync_logs()

        failed_dependencies = [d for d in dependencies if d.state.is_failed()]
        if job.state.is_failed() or failed_dependencies:
            info = job.info
            # TODO: put logs here
            name = info.name if info else "<unknown>"
            description = info.description if info else "<unknown>"

            failed_jobs = copy.copy(failed_dependencies)
            if job.state.is_failed():
                failed_jobs.insert(0, job)

            raise ClusterNavigatorException(f"[{job.job_id}] {description or name} failed", failed_jobs)
        return job

    def exec(self, job_id: str, command: str, check: bool = True, timeout_s: typing.Optional[int] = None):
        job = self.get_job(job_id)
        if job.state.is_done():
            raise ClusterNavigatorException(
                f"Could not execute command {command} on job {job_id} while it is in state {job.state}"
            )

        command = self._exec_command.format(job_id=job_id, command=shlex.quote(command))
        return execute_command_and_get_result(command, check=check, timeout_s=timeout_s)

    def cancel(self, job_id: str, check: bool = True):
        job = self.get_job(job_id)
        if not job.state.is_done():
            command = self._cancel_command.format(job_id=job_id)
            execute_command_and_get_result(command, check=check, timeout_s=self._command_timeout_s)
