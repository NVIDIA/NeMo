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

from pytest import mark, param, raises

from nemo.tron.utils.instantiate_utils import (InstantiationException,
                                               instantiate)
from tests.tron.utils.instantiate import ArgsClass

#######################
# non-recursive tests #
#######################


@mark.parametrize(
    ("cfg", "expected"),
    [
        param(
            {"_target_": "tests.tron.utils.instantiate.ArgsClass"},
            ArgsClass(),
            id="config:no_params",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.ArgsClass", "_args_": [1]},
            ArgsClass(1),
            id="config:args_only",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.ArgsClass",
                "_args_": [1],
                "foo": 10,
            },
            ArgsClass(1, foo=10),
            id="config:args+kwargs",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.ArgsClass",
                "foo": 10,
            },
            ArgsClass(foo=10),
            id="config:kwargs_only",
        ),
    ],
)
def test_instantiate_args_kwargs(cfg: Any, expected: Any) -> None:
    assert instantiate(cfg) == expected


@mark.parametrize(
    "cfg, msg",
    [
        param(
            {
                "_target_": "tests.tron.utils.instantiate.ArgsClass",
                "_args_": {"foo": "bar"},
            },
            "Unsupported _args_ type: 'dict'",
            id="unsupported-args-type",
        ),
        param(
            {
                "foo": {
                    "_target_": "tests.tron.utils.instantiate.ArgsClass",
                    "_args_": {"foo": "bar"},
                }
            },
            "Unsupported _args_ type: 'dict'",
            id="unsupported-args-type-nested",
        ),
    ],
)
def test_instantiate_unsupported_args_type(cfg: Any, msg: str) -> None:
    with raises(
        InstantiationException,
        match=msg,
    ):
        instantiate(cfg)


@mark.parametrize(
    ("cfg", "expected"),
    [
        param(
            {
                "_target_": "tests.tron.utils.instantiate.ArgsClass",
                "_args_": ["${.1}", 2],
            },
            ArgsClass(2, 2),
            id="config:args_only",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.ArgsClass",
                "_args_": [1],
                "foo": "${._args_}",
            },
            ArgsClass(1, foo=[1]),
            id="config:args+kwargs",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.ArgsClass",
                "foo": "${._target_}",
            },
            ArgsClass(foo="tests.tron.utils.instantiate.ArgsClass"),
            id="config:kwargs_only)",
        ),
    ],
)
def test_instantiate_args_kwargs_with_interpolation(cfg: Any, expected: Any) -> None:
    assert instantiate(cfg) == expected


@mark.parametrize(
    ("cfg", "args_override", "kwargs_override", "expected"),
    [
        param(
            {"_target_": "tests.tron.utils.instantiate.ArgsClass", "_args_": [1]},
            [2],
            {},
            ArgsClass(2),
            id="direct_args",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.ArgsClass", "_args_": [1]},
            [],
            {"_args_": [2]},
            ArgsClass(2),
            id="indirect_args",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.ArgsClass", "_args_": [1]},
            [],
            {"foo": 10},
            ArgsClass(1, foo=10),
            id="kwargs",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.ArgsClass", "_args_": [1]},
            [],
            {"foo": 10, "_args_": [2]},
            ArgsClass(2, foo=10),
            id="kwargs+indirect_args",
        ),
    ],
)
def test_instantiate_args_kwargs_with_override(
    cfg: Any, args_override: Any, expected: Any, kwargs_override: Any
) -> None:
    assert instantiate(cfg, *args_override, **kwargs_override) == expected


###################
# recursive tests #
###################
@mark.parametrize(
    ("cfg", "expected"),
    [
        param(
            {
                "_target_": "tests.tron.utils.instantiate.ArgsClass",
                "child": {"_target_": "tests.tron.utils.instantiate.ArgsClass"},
            },
            ArgsClass(child=ArgsClass()),
            id="config:no_params",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.ArgsClass",
                "_args_": [1],
                "child": {
                    "_target_": "tests.tron.utils.instantiate.ArgsClass",
                    "_args_": [2],
                },
            },
            ArgsClass(1, child=ArgsClass(2)),
            id="config:args_only",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.ArgsClass",
                "_args_": [1],
                "foo": 10,
                "child": {
                    "_target_": "tests.tron.utils.instantiate.ArgsClass",
                    "_args_": [2],
                },
            },
            ArgsClass(1, foo=10, child=ArgsClass(2)),
            id="config:args+kwargs",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.ArgsClass",
                "child": {
                    "_target_": "tests.tron.utils.instantiate.ArgsClass",
                    "_args_": [2],
                },
                "foo": 10,
            },
            ArgsClass(foo=10, child=ArgsClass(2)),
            id="config:kwargs_only",
        ),
    ],
)
def test_recursive_instantiate_args_kwargs(cfg: Any, expected: Any) -> None:
    assert instantiate(cfg) == expected


@mark.parametrize(
    ("cfg", "args_override", "kwargs_override", "expected"),
    [
        param(
            {
                "_target_": "tests.tron.utils.instantiate.ArgsClass",
                "child": {"_target_": "tests.tron.utils.instantiate.ArgsClass"},
            },
            [1],
            {},
            ArgsClass(1, child=ArgsClass()),
            id="direct_args_not_in_nested",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.ArgsClass",
                "_args_": [1],
                "child": {
                    "_target_": "tests.tron.utils.instantiate.ArgsClass",
                    "_args_": [2],
                },
            },
            [],
            {"_args_": [3], "child": {"_args_": [4]}},
            ArgsClass(3, child=ArgsClass(4)),
            id="indirect_args",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.ArgsClass",
                "_args_": [1],
                "child": {
                    "_target_": "tests.tron.utils.instantiate.ArgsClass",
                    "_args_": [2],
                },
            },
            [],
            {"foo": 10},
            ArgsClass(1, foo=10, child=ArgsClass(2)),
            id="kwargs",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.ArgsClass",
                "_args_": [1],
                "child": {
                    "_target_": "tests.tron.utils.instantiate.ArgsClass",
                    "_args_": [2],
                },
            },
            [],
            {"foo": 10, "_args_": [3], "child": {"_args_": [4]}},
            ArgsClass(3, foo=10, child=ArgsClass(4)),
            id="kwargs+indirect_args",
        ),
    ],
)
def test_recursive_instantiate_args_kwargs_with_override(
    cfg: Any, args_override: Any, expected: Any, kwargs_override: Any
) -> None:
    assert instantiate(cfg, *args_override, **kwargs_override) == expected
