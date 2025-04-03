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

from __future__ import annotations

import collections
import collections.abc
import functools
import json
import random
import time
from contextlib import ContextDecorator
from typing import Any, Callable, TypeVar

import numpy as np
import termcolor
import torch

from cosmos1.utils import distributed, log


def to(
    data: Any,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    memory_format: torch.memory_format = torch.preserve_format,
) -> Any:
    """Recursively cast data into the specified device, dtype, and/or memory_format.

    The input data can be a tensor, a list of tensors, a dict of tensors.
    See the documentation for torch.Tensor.to() for details.

    Args:
        data (Any): Input data.
        device (str | torch.device): GPU device (default: None).
        dtype (torch.dtype): data type (default: None).
        memory_format (torch.memory_format): memory organization format (default: torch.preserve_format).

    Returns:
        data (Any): Data cast to the specified device, dtype, and/or memory_format.
    """
    assert (
        device is not None or dtype is not None or memory_format is not None
    ), "at least one of device, dtype, memory_format should be specified"
    if isinstance(data, torch.Tensor):
        is_cpu = (isinstance(device, str) and device == "cpu") or (
            isinstance(device, torch.device) and device.type == "cpu"
        )
        data = data.to(
            device=device,
            dtype=dtype,
            memory_format=memory_format,
            non_blocking=(not is_cpu),
        )
        return data
    elif isinstance(data, collections.abc.Mapping):
        return type(data)({key: to(data[key], device=device, dtype=dtype, memory_format=memory_format) for key in data})
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        return type(data)([to(elem, device=device, dtype=dtype, memory_format=memory_format) for elem in data])
    else:
        return data


def serialize(data: Any) -> Any:
    """Serialize data by hierarchically traversing through iterables.

    Args:
        data (Any): Input data.

    Returns:
        data (Any): Serialized data.
    """
    if isinstance(data, collections.abc.Mapping):
        return type(data)({key: serialize(data[key]) for key in data})
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        return type(data)([serialize(elem) for elem in data])
    else:
        try:
            json.dumps(data)
        except TypeError:
            data = str(data)
        return data


def set_random_seed(seed: int, by_rank: bool = False) -> None:
    """Set random seed. This includes random, numpy, Pytorch.

    Args:
        seed (int): Random seed.
        by_rank (bool): if true, each GPU will use a different random seed.
    """
    if by_rank:
        seed += distributed.get_rank()
    log.info(f"Using random seed {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # sets seed on the current CPU & all GPUs


def arch_invariant_rand(
    shape: List[int] | Tuple[int], dtype: torch.dtype, device: str | torch.device, seed: int | None = None
):
    """Produce a GPU-architecture-invariant randomized Torch tensor.

    Args:
        shape (list or tuple of ints): Output tensor shape.
        dtype (torch.dtype): Output tensor type.
        device (torch.device): Device holding the output.
        seed (int): Optional randomization seed.

    Returns:
        tensor (torch.tensor): Randomly-generated tensor.
    """
    # Create a random number generator, optionally seeded
    rng = np.random.RandomState(seed)

    # # Generate random numbers using the generator
    random_array = rng.standard_normal(shape).astype(np.float32)  # Use standard_normal for normal distribution

    # Convert to torch tensor and return
    return torch.from_numpy(random_array).to(dtype=dtype, device=device)


T = TypeVar("T", bound=Callable[..., Any])


class timer(ContextDecorator):  # noqa: N801
    """Simple timer for timing the execution of code.

    It can be used as either a context manager or a function decorator. The timing result will be logged upon exit.

    Example:
        def func_a():
            time.sleep(1)
        with timer("func_a"):
            func_a()

        @timer("func_b)
        def func_b():
            time.sleep(1)
        func_b()
    """

    def __init__(self, context: str, debug: bool = False):
        self.context = context
        self.debug = debug

    def __enter__(self) -> None:
        self.tic = time.time()

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        time_spent = time.time() - self.tic
        if self.debug:
            log.debug(f"Time spent on {self.context}: {time_spent:.4f} seconds")
        else:
            log.debug(f"Time spent on {self.context}: {time_spent:.4f} seconds")

    def __call__(self, func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # noqa: ANN202
            tic = time.time()
            result = func(*args, **kwargs)
            time_spent = time.time() - tic
            if self.debug:
                log.debug(f"Time spent on {self.context}: {time_spent:.4f} seconds")
            else:
                log.debug(f"Time spent on {self.context}: {time_spent:.4f} seconds")
            return result

        return wrapper  # type: ignore


class Color:
    """A convenience class to colorize strings in the console.

    Example:
        import
        print("This is {Color.red('important')}.")
    """

    @staticmethod
    def red(x: str) -> str:
        return termcolor.colored(str(x), color="red")

    @staticmethod
    def green(x: str) -> str:
        return termcolor.colored(str(x), color="green")

    @staticmethod
    def cyan(x: str) -> str:
        return termcolor.colored(str(x), color="cyan")

    @staticmethod
    def yellow(x: str) -> str:
        return termcolor.colored(str(x), color="yellow")
