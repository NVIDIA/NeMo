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
import pathlib
import re
import time
import typing

from .cluster.job import JobStatus
from .utils import MIN2S

LOGGER = logging.getLogger(__name__)

DEFAULT_HTTP_PORT = 8000
DEFAULT_GRPC_PORT = 8001
DEFAULT_METRIC_PORT = 8002
DEFAULT_MAX_WAIT_FOR_TRITONSERVER_SCHEDULE_S = 30 * MIN2S


@dataclasses.dataclass
class Variant:
    model_name: str
    max_batch_size: typing.Optional[int] = None
    tensor_parallel_size: typing.Optional[int] = None
    pipeline_parallel_size: typing.Optional[int] = None
    input_output_lengths_pair: typing.Optional[typing.Tuple[int, int]] = None
    is_half: typing.Optional[bool] = None
    vocab_size: typing.Optional[int] = None
    end_id: typing.Optional[int] = None

    @property
    def extended_name(self):
        io = (
            f"{self.input_output_lengths_pair[0]}_{self.input_output_lengths_pair[1]}"
            if self.input_output_lengths_pair
            else None
        )
        parameters_to_be_included_in_name = [
            ("io", io),
            ("half", int(self.is_half) if self.is_half is not None else None),
            ("pp", self.pipeline_parallel_size),
            ("tp", self.tensor_parallel_size),
            ("mbs", self.max_batch_size),
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
        vocab_size = _search_backend_param("vocab_size")
        end_id = _search_backend_param("end_id")
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
            vocab_size=_cast_if_not_none(vocab_size, int),
            end_id=_cast_if_not_none(end_id, int),
        )

    @classmethod
    def from_triton_model_repository(cls, triton_model_repository_path: pathlib.Path):
        triton_config_paths = list(triton_model_repository_path.rglob("config.pbtxt"))
        if len(triton_config_paths) > 1:
            raise ValueError(f"More than single config.pbtxt in {triton_model_repository_path}")
        elif not triton_config_paths:
            raise ValueError(f"Could not find config.pbtxt in {triton_model_repository_path}")
        triton_config_path = triton_config_paths[0]
        return cls.from_triton_config(triton_config_path)


class TritonServerSet:
    def __init__(self, job):
        self._job = job

    @property
    def grpc_endpoints(self) -> typing.List[str]:
        return [self._job.get_endpoint_url("grpc", DEFAULT_GRPC_PORT)]

    @property
    def http_endpoints(self) -> typing.List[str]:
        return [self._job.get_endpoint_url("http", DEFAULT_HTTP_PORT)]

    @property
    def metric_endpoints(self) -> typing.List[str]:
        return [self._job.get_endpoint_url("http", DEFAULT_METRIC_PORT) + "/metrics"]

    def wait_until_job_is_running_or_done(self, timeout_s: int = DEFAULT_MAX_WAIT_FOR_TRITONSERVER_SCHEDULE_S):
        step_s = 5
        while self.state != JobStatus.RUNNING and not self.state.is_done() and timeout_s > 0:
            time.sleep(step_s)
            self._job.update_info()
            timeout_s -= step_s

        if timeout_s <= 0:
            raise TimeoutError(f"[{self._job.job_id}] Could not schedule TritonServerSet job")

    @property
    def state(self):
        return self._job.state
