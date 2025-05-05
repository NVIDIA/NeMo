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

from typing import Any


class PosOnlyArgsClass:
    def __init__(self, a: Any, b: Any, /, **kwargs: Any) -> None:
        assert isinstance(kwargs, dict)
        self.a = a
        self.b = b
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f"{self.a=},{self.b},{self.kwargs=}"

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, PosOnlyArgsClass):
            return self.a == other.a and self.b == other.b and self.kwargs == other.kwargs
        else:
            return NotImplemented
