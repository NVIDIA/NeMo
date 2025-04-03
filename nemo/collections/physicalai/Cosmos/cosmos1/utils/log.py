# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import os
import sys
from typing import Any, Optional

import torch.distributed as dist
from loguru._logger import Core, Logger

RANK0_ONLY = True
LEVEL = os.environ.get("LOGURU_LEVEL", "INFO")

logger = Logger(
    core=Core(),
    exception=None,
    depth=1,
    record=False,
    lazy=False,
    colors=False,
    raw=False,
    capture=True,
    patchers=[],
    extra={},
)

atexit.register(logger.remove)


def _add_relative_path(record: dict[str, Any]) -> None:
    start = os.getcwd()
    record["extra"]["relative_path"] = os.path.relpath(record["file"].path, start)


*options, _, extra = logger._options  # type: ignore
logger._options = tuple([*options, [_add_relative_path], extra])  # type: ignore


def init_loguru_stdout() -> None:
    logger.remove()
    machine_format = get_machine_format()
    message_format = get_message_format()
    logger.add(
        sys.stdout,
        level=LEVEL,
        format="[<green>{time:MM-DD HH:mm:ss}</green>|" f"{machine_format}" f"{message_format}",
        filter=_rank0_only_filter,
    )


def get_machine_format() -> str:
    node_id = os.environ.get("NGC_ARRAY_INDEX", "0")
    num_nodes = int(os.environ.get("NGC_ARRAY_SIZE", "1"))
    machine_format = ""
    rank = 0
    if dist.is_available():
        if not RANK0_ONLY and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            machine_format = (
                f"<red>[Node{node_id:<3}/{num_nodes:<3}][RANK{rank:<5}/{world_size:<5}]" + "[{process.name:<8}]</red>| "
            )
    return machine_format


def get_message_format() -> str:
    message_format = "<level>{level}</level>|<cyan>{extra[relative_path]}:{line}:{function}</cyan>] {message}"
    return message_format


def _rank0_only_filter(record: Any) -> bool:
    is_rank0 = record["extra"].get("rank0_only", True)
    if _get_rank() == 0 and is_rank0:
        return True
    if not is_rank0:
        record["message"] = f"[RANK {_get_rank()}]" + record["message"]
    return not is_rank0


def trace(message: str, rank0_only: bool = True) -> None:
    logger.opt(depth=1).bind(rank0_only=rank0_only).trace(message)


def debug(message: str, rank0_only: bool = True) -> None:
    logger.opt(depth=1).bind(rank0_only=rank0_only).debug(message)


def info(message: str, rank0_only: bool = True) -> None:
    logger.opt(depth=1).bind(rank0_only=rank0_only).info(message)


def success(message: str, rank0_only: bool = True) -> None:
    logger.opt(depth=1).bind(rank0_only=rank0_only).success(message)


def warning(message: str, rank0_only: bool = True) -> None:
    logger.opt(depth=1).bind(rank0_only=rank0_only).warning(message)


def error(message: str, rank0_only: bool = True) -> None:
    logger.opt(depth=1).bind(rank0_only=rank0_only).error(message)


def critical(message: str, rank0_only: bool = True) -> None:
    logger.opt(depth=1).bind(rank0_only=rank0_only).critical(message)


def exception(message: str, rank0_only: bool = True) -> None:
    logger.opt(depth=1).bind(rank0_only=rank0_only).exception(message)


def _get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """Get the rank (GPU device) of the worker.

    Returns:
        rank (int): The rank of the worker.
    """
    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group)
    return rank


# Execute at import time.
init_loguru_stdout()
