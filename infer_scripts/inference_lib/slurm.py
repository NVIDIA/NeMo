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
import atexit
import enum
import logging
import os
import pathlib
import re
import shlex
import time
import typing
import typing as tp
from pathlib import Path
from typing import Union

import submitit
from submitit.core import utils

from .utils import config_logger

LOGGER = logging.getLogger(__name__)

DEFAULT_JOB_NAME_PREFIX = "bignlp-"
WCKEY = "BigNLP"
TRITON_MODEL_REPOSITORY = "/triton-model-repository"


class ContainerImageType(enum.Enum):
    TRAINING = "training"
    INFERENCE = "inference"


class DirToMount(typing.NamedTuple):
    src: pathlib.Path
    dst: pathlib.Path
    readonly: bool


"""
Possible submitit slurm executor parameters
'gpus_per_task', 'gpus_per_node', 'num_gpus', 'cpus_per_task', 'cpus_per_gpu',
'mem_per_gpu', 'mem_per_cpu', 'mem',
'gres', 'constraint', 'exclude', 'exclusive'
'nodes', 'ntasks_per_node',
'array_parallelism',
'job_name', 'wckey', 'comment', 'partition', 'time',
'stderr_to_stdout',
'srun_args', 'additional_parameters', 'setup'
'signal_delay_s', 'qos'
"""


def get_common_slurm_parameters_new(
    *,
    cluster_config: typing.Dict[str, typing.Any],
    container_image_type: ContainerImageType,
    dirs_to_mount: typing.Optional[typing.List[typing.Tuple[pathlib.Path, pathlib.Path]]] = None,
):
    from .inference import get_dirs_to_mount_new

    slurm_config: typing.Dict[str, typing.Any] = cluster_config["slurm"]
    env_config = cluster_config["env"]

    def _rewrite_docker_image(docker_image_):
        if "#" not in docker_image_:
            docker_image_parts = docker_image_.split("/")
            docker_image_ = f"{docker_image_parts[0]}#{'/'.join(docker_image_parts[1:])}"
        return docker_image_

    container_image = {
        ContainerImageType.TRAINING: _rewrite_docker_image(env_config["pyxis_training_container_image"]),
        ContainerImageType.INFERENCE: _rewrite_docker_image(env_config["pyxis_inference_container_image"]),
    }[container_image_type]

    dirs_to_mount = get_dirs_to_mount_new(dirs_to_mount or [], container_image_type=container_image_type)
    container_mounts = ",".join(
        f"{d.src.absolute()}:{d.dst.absolute()}{':ro' if d.readonly else ''}" for d in (dirs_to_mount or [])
    )

    sbatch_args = {k: v for k, v in slurm_config["sbatch_parameters"].items() if v}
    partition = sbatch_args.pop("partition")
    srun_args = slurm_config.get("srun_args", []) + [
        "--container-workdir",
        env_config["pyxis_container_workdir"],
        "--container-image",
        container_image,
        "--container-mounts",
        container_mounts,
    ]

    return {
        "partition": partition,
        "wckey": WCKEY,
        "stderr_to_stdout": True,
        "additional_parameters": sbatch_args,
        "srun_args": srun_args,
    }


def get_common_slurm_parameters(
    *,
    cluster_config: typing.Dict[str, typing.Any],
    container_image_type: ContainerImageType,
    dirs_to_mount: typing.Optional[typing.List[DirToMount]] = None,
):
    slurm_config: typing.Dict[str, typing.Any] = cluster_config["slurm"]
    env_config = cluster_config["env"]

    def _rewrite_docker_image(docker_image_):
        if "#" not in docker_image_:
            docker_image_parts = docker_image_.split("/")
            docker_image_ = f"{docker_image_parts[0]}#{'/'.join(docker_image_parts[1:])}"
        return docker_image_

    container_image = {
        ContainerImageType.TRAINING: _rewrite_docker_image(env_config["pyxis_training_container_image"]),
        ContainerImageType.INFERENCE: _rewrite_docker_image(env_config["pyxis_inference_container_image"]),
    }[container_image_type]
    container_mounts = ",".join(
        f"{d.src.absolute()}:{d.dst.absolute()}{':ro' if d.readonly else ''}" for d in (dirs_to_mount or [])
    )

    sbatch_args = {k: v for k, v in slurm_config["sbatch_parameters"].items() if v}
    partition = sbatch_args.pop("partition")
    srun_args = slurm_config.get("srun_args", []) + [
        "--container-workdir",
        env_config["pyxis_container_workdir"],
        "--container-image",
        container_image,
        "--container-mounts",
        container_mounts,
    ]

    return {
        "partition": partition,
        "wckey": WCKEY,
        "stderr_to_stdout": True,
        "additional_parameters": sbatch_args,
        "srun_args": srun_args,
    }


def get_cluster_suffix() -> str:
    env = os.environ
    job_id = env.get("SLURM_JOB_ID")
    rank_id = env.get("SLURM_LOCALID")
    step_id = env.get("SLURM_STEP_ID")
    return f"{job_id}_{rank_id}_{step_id}"


def parse_node_list(node_list: str) -> typing.List[str]:
    # prm-dgx-[16-17]
    if "[" in node_list:
        final_idxes = []

        range_match = re.match(r"(?P<base>[\w-]+)\[(?P<range>.*)\]", node_list)
        groups = range_match.groupdict()
        idxes = groups["range"].split(",")
        for idx in idxes:
            if "-" in idx:
                start, stop = idx.split("-")
                nchars = len(start)
                start, stop = int(start), int(stop) + 1
                final_idxes.extend([f"{idx:0{nchars}d}" for idx in range(start, stop)])
            else:
                final_idxes.append(idx)
        parsed_node_list = [f"{groups['base']}{idx}" for idx in final_idxes]
    else:
        parsed_node_list = [node_list]

    return parsed_node_list


class PyxisExecutor(submitit.SlurmExecutor):
    def __init__(self, folder: Union[Path, str], max_num_timeout: int = 3) -> None:
        super().__init__(folder, max_num_timeout)

        self._submitted_jobs = []
        atexit.register(self._close_all_jobs)

    def _close_all_jobs(self):
        for job in self._submitted_jobs:
            job.cancel()

    def _internal_process_submissions(self, delayed_submissions: tp.List[utils.DelayedSubmission]):
        jobs = super()._internal_process_submissions(delayed_submissions)
        self._submitted_jobs.extend(jobs)
        return jobs

    @property
    def _submitit_command_str(self) -> str:
        # use default enroot container python interpreter instead of the one used to submit job
        return " ".join(["python3 -u -m submitit.core._submit", shlex.quote(str(self.folder))])


class PyxisInfoWatcher(submitit.slurm.slurm.SlurmInfoWatcher):
    def _make_command(self) -> typing.Optional[typing.List[str]]:
        # asking for array id will return all status
        # on the other end, asking for each and every one of them individually takes a huge amount of time
        to_check = {x.split("_")[0] for x in self._registered - self._finished}
        if not to_check:
            return None
        command = ["sacct", "-o", "JobID,JobName,Comment,State,NodeList", "--parsable2"]
        for jid in to_check:
            command.extend(["-j", str(jid)])
        return command

    def update(self) -> None:
        copy_of_info_dict = self._info_dict.copy()
        super().update()
        for job_id, info in self._info_dict.items():
            previous_info = copy_of_info_dict.get(job_id, None)
            current_state = info.get("State", "UNKNOWN")
            current_nodes = info.get("NodeList", "UNKNOWN")
            previous_state = previous_info.get("State", "UNKNOWN") if previous_info else None
            if job_id not in copy_of_info_dict or current_state != previous_state:
                LOGGER.info(
                    f"[{job_id}/{info.get('JobName', '<UNKNOWN_NAME>')}] "
                    f"state changed to {current_state} on {current_nodes}"
                )


class PyxisTritonExecutor(submitit.SlurmExecutor):
    def __init__(self, folder: Union[Path, str], max_num_timeout: int = 3) -> None:
        super().__init__(folder, max_num_timeout)
        self._submitted_jobs = []
        atexit.register(self._close_all_jobs)

    def _close_all_jobs(self):
        for job in self._submitted_jobs:
            job.cancel()

    def _internal_process_submissions(self, delayed_submissions: tp.List[utils.DelayedSubmission]):
        jobs = super()._internal_process_submissions(delayed_submissions)
        self._submitted_jobs.extend(jobs)
        return jobs

    @property
    def _submitit_command_str(self) -> str:
        return f"tritonserver --model-repository={TRITON_MODEL_REPOSITORY}"


def init_job_env(verbose: bool = False):
    import pickle

    # to be able to read results on py3.6+ created on py3.8+
    pickle.HIGHEST_PROTOCOL = 4

    config_logger(verbose)


def setup_job(job):
    # WAR for issue on prom - 1st get_state returns COMPLETED state
    timeout_s = 5
    while job.watcher.get_state(job.job_id, mode="force") == "COMPLETED" and timeout_s > 0:
        job.watcher._finished = set()
        time.sleep(1)
        timeout_s -= 1

    job.cancel_at_deletion()
