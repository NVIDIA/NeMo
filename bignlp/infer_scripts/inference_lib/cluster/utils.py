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
import json
import logging
import pathlib
import re
import subprocess
import textwrap
import time
import typing
import uuid

LOGGER = logging.getLogger(__name__)


class ClusterNavigatorPathsGenerator:
    _base_name_pattern = r"submission_(?P<job_id>\w+)(?P<suffix>\.\w+)"
    _base_name_format = "submission_{job_id}{suffix}"

    def __init__(self, cluster_dir: pathlib.Path):
        self._cluster_dir = cluster_dir

    @property
    def cluster_dir(self) -> pathlib.Path:
        return self._cluster_dir

    def get_submission_file_path(self, job_id: str, suffix: str):
        return self._cluster_dir / self._name_with_job_id_and_suffix(job_id=job_id, suffix=suffix)

    def get_log_path(self, job_id: str):
        return self._cluster_dir / self._name_with_job_id_and_suffix(job_id=job_id, suffix=".out")

    def get_temporary_file_path(self, suffix: str) -> pathlib.Path:
        while True:
            random_job_id = uuid.uuid4().hex
            temporary_file_path = self.cluster_dir / self._name_with_job_id_and_suffix(random_job_id, suffix)
            if not temporary_file_path.exists():
                break
        return temporary_file_path

    def replace_job_id_and_rename_file(self, src_file_path: pathlib.Path, job_id: str):
        match = re.match(self._base_name_pattern, src_file_path.name)
        match = match.groupdict()
        new_filename = self._name_with_job_id_and_suffix(job_id, match.get("suffix"))
        new_path = src_file_path.parent / new_filename
        src_file_path.rename(new_path)
        return new_path

    def _name_with_job_id_and_suffix(self, job_id: str, suffix: str) -> str:
        return self._base_name_format.format(job_id=job_id, suffix=suffix)


def execute_command_and_get_result(command: str, check: bool = True, timeout_s: typing.Optional[int] = None):
    started_at_s = time.time()
    try:
        completed_process = subprocess.run(
            command,
            shell=True,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            timeout=timeout_s,
        )
    except subprocess.CalledProcessError as e:
        from .executor import ClusterNavigatorException

        raise ClusterNavigatorException(f"{e}. Output:\n\n{textwrap.indent(e.stdout, '    ')}")
    LOGGER.debug(f"executed: {command} in {time.time() - started_at_s:0.1f}s")
    result = None
    if completed_process.stdout:
        result = completed_process.stdout
    return result


def execute_command_and_get_json_result(command: str, check: bool = True, timeout_s: typing.Optional[int] = None):
    result = execute_command_and_get_result(command, check, timeout_s)
    if result:
        result = json.loads(result)
    return result
