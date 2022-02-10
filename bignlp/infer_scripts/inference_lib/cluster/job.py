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
import dataclasses
import enum
import logging
import pathlib
import time
import typing

LOGGER = logging.getLogger(__name__)


class MountMode(enum.Enum):
    RO = "ro"
    RW = "rw"


class JobStatus(enum.Enum):
    QUEUED = "QUEUED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    FINISHED_SUCCESS = "FINISHED_SUCCESS"
    KILLED_BY_SYSTEM = "KILLED_BY_SYSTEM"
    KILLED_BY_USER = "KILLED_BY_USER"
    PREEMPTED = "PREEMPTED"
    UNKNOWN = "UNKNOWN"

    def is_done(self):
        return self in [
            JobStatus.FAILED,
            JobStatus.FINISHED_SUCCESS,
            JobStatus.KILLED_BY_USER,
            JobStatus.KILLED_BY_SYSTEM,
        ]

    def is_failed(self):
        return self in [JobStatus.FAILED, JobStatus.KILLED_BY_SYSTEM]


@dataclasses.dataclass(frozen=True)
class DirToMount:
    src: typing.Union[str, pathlib.Path]
    dst: pathlib.Path
    mode: MountMode = MountMode.RW

    def __str__(self):
        src = self.src.resolve().absolute().as_posix() if isinstance(self.src, pathlib.Path) else self.src
        dst = self.dst.resolve().absolute().as_posix()
        return f"{src}:{dst}:{self.mode.value.upper()}"


@dataclasses.dataclass
class RemoteFunctionDefinition:
    fn: typing.Callable
    args: typing.Optional[typing.List[typing.Any]] = None
    kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None


@dataclasses.dataclass
class JobDefinition:
    name: str
    description: str

    max_time_s: int
    container_image: str
    commands: typing.Union[typing.List[str], RemoteFunctionDefinition]

    workdir_path: typing.Optional[pathlib.Path] = None
    directories_to_mount: typing.Optional[
        typing.Union[typing.List[typing.Union[str, pathlib.Path]], typing.List[DirToMount]]
    ] = None
    ports: typing.Optional[typing.List[int]] = None

    tasks_number: typing.Optional[int] = None
    tasks_number_per_node: typing.Optional[typing.Union[int, typing.Dict[str, int]]] = None
    gpus_number_per_task: typing.Optional[int] = None
    cpus_number_per_task: typing.Optional[int] = None
    mem_number_gb_per_task: typing.Optional[int] = None

    setup_commands: typing.Optional[typing.List[str]] = None
    clean_commands: typing.Optional[typing.List[str]] = None

    def __post_init__(self):
        if self.tasks_number_per_node and self.tasks_number % self.tasks_number_per_node != 0:
            raise ValueError(
                f"Number of tasks ({self.tasks_number}) should be "
                f"multiple of tasks_number_per_node ({self.tasks_number_per_node})"
            )


@dataclasses.dataclass(frozen=True)
class JobInfo:
    job_id: str
    name: str
    description: str
    state: JobStatus


class BaseJob(abc.ABC):
    def __init__(self, job_id: str, executor):
        self._job_id = job_id
        self._executor = executor
        self._last_state = None

        info = self.info
        if info:
            self._last_state = info.state

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def info(self) -> typing.Optional[JobInfo]:
        return self._executor.get_info(self._job_id)

    @property
    def state(self) -> JobStatus:
        new_state = JobStatus.UNKNOWN
        info = self.info
        if info:
            new_state = info.state
            if (self._last_state is not None and new_state != self._last_state) or self._last_state is None:
                if self._last_state:
                    state_change = f"{self._last_state} -> {new_state}"
                else:
                    state_change = new_state
                LOGGER.info(f"[{self.job_id}] {info.description or info.name} (job state: {state_change})")
            self._last_state = new_state
        return new_state

    @abc.abstractmethod
    def get_endpoint_url(self, scheme: str, port: int) -> str:
        pass

    def exec(self, command: str, check: bool = True, timeout_s: typing.Optional[int] = None):
        return self._executor.exec(self._job_id, command, check, timeout_s)

    def cancel(self):
        self._executor.cancel(self._job_id)

    def wait(self, dependencies: typing.Optional[typing.List] = None):
        dependencies = dependencies or []
        failed_dependencies = []
        while not self.state.is_done() and not failed_dependencies:
            time.sleep(1)
            failed_dependencies = [d for d in dependencies if d.state.is_failed()]

    def update_info(self) -> typing.Optional[JobInfo]:
        return self._executor.get_info(self._job_id, force_update=True)

    @property
    def log_path(self) -> pathlib.Path:
        return self._executor.paths.get_log_path(job_id=self._job_id)

    def sync_logs(self):
        pass
