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

from pytest import mark, param

from nemo.tron.utils.instantiate_utils import instantiate

from .positional_only import PosOnlyArgsClass


@mark.parametrize(
    ("cfg", "args", "expected"),
    [
        param(
            {
                "_target_": "utils.instantiate.positional_only.PosOnlyArgsClass",
                "_args_": [1, 2],
            },
            [],
            PosOnlyArgsClass(1, 2),
            id="pos_only_in_config",
        ),
        param(
            {
                "_target_": "utils.instantiate.positional_only.PosOnlyArgsClass",
            },
            [1, 2],
            PosOnlyArgsClass(1, 2),
            id="pos_only_in_override",
        ),
        param(
            {
                "_target_": "utils.instantiate.positional_only.PosOnlyArgsClass",
                "_args_": [1, 2],
            },
            [3, 4],
            PosOnlyArgsClass(3, 4),
            id="pos_only_in_both",
        ),
    ],
)
def test_positional_only_arguments(cfg: Any, args: Any, expected: Any) -> None:
    assert instantiate(cfg, *args) == expected
