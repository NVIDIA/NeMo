# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import functools
import logging
import warnings
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

# Define a TypeVar for generic return types
R = TypeVar('R')


def experimental_fn(func: Callable[..., R]) -> Callable[..., R]:
    """Decorator to mark a function as experimental and issue a warning upon its call."""
    warning_message = (
        f"Function '{func.__name__}' is experimental. APIs in this module are subject to change without notice."
    )

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        warnings.warn(warning_message, stacklevel=2)
        return func(*args, **kwargs)

    return wrapper
