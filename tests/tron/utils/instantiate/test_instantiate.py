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

import copy
import logging
import os
import pickle
import re
from functools import partial
from textwrap import dedent
from typing import Any, Callable, Dict

import pytest
import yaml
from omegaconf import DictConfig, ListConfig, MissingMandatoryValue, OmegaConf
from pytest import fixture, mark, param, raises

from nemo.tron.utils.instantiate_utils import (
    InstantiationException,
    InstantiationMode,
    _convert_node,
    _convert_target_to_string,
    _locate,
    instantiate,
    instantiate_node,
)
from nemo.tron.utils.yaml_utils import safe_yaml_representers
from tests.tron.utils.instantiate import (
    AClass,
    Adam,
    AdamConf,
    AnotherClass,
    ArgsClass,
    ASubclass,
    BClass,
    CenterCrop,
    CenterCropConf,
    Compose,
    ComposeConf,
    IllegalType,
    KeywordsInParamsClass,
    Mapping,
    MappingConf,
    NestingClass,
    OuterClass,
    Parameters,
    Rotation,
    RotationConf,
    SimpleClass,
    TargetWithInstantiateInInit,
    Tree,
    TreeConf,
    UntypedPassthroughClass,
    UntypedPassthroughConf,
    User,
    add_values,
    module_function,
    module_function2,
    partial_equal,
)


@fixture(
    params=[
        lambda cfg: copy.deepcopy(cfg),
        lambda cfg: OmegaConf.create(cfg),
    ],
    ids=[
        "dict",
        "dict_config",
    ],
)
def config(request: Any, src: Any) -> Any:
    config = request.param(src)
    cfg_copy = copy.deepcopy(config)
    yield config
    assert config == cfg_copy


@mark.parametrize(
    "src, passthrough, expected",
    [
        param(
            {
                "_target_": "tests.tron.utils.instantiate.AClass",
                "a": 10,
                "b": 20,
                "c": 30,
                "d": 40,
            },
            {},
            AClass(10, 20, 30, 40),
            id="class",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.AClass",
                "_partial_": True,
                "a": 10,
                "b": 20,
                "c": 30,
            },
            {},
            partial(AClass, a=10, b=20, c=30),
            id="class+partial",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.AClass",
                "_partial_": True,
                "a": "???",
                "b": 20,
                "c": 30,
            },
            {},
            partial(AClass, b=20, c=30),
            id="class+partial+missing",
        ),
        param(
            [
                {
                    "_target_": "tests.tron.utils.instantiate.AClass",
                    "_partial_": True,
                    "a": 10,
                    "b": 20,
                    "c": 30,
                },
                {
                    "_target_": "tests.tron.utils.instantiate.BClass",
                    "a": 50,
                    "b": 60,
                    "c": 70,
                },
            ],
            {},
            [partial(AClass, a=10, b=20, c=30), BClass(a=50, b=60, c=70)],
            id="list_of_partial_class",
        ),
        param(
            [
                {
                    "_target_": "tests.tron.utils.instantiate.AClass",
                    "_partial_": True,
                    "a": "???",
                    "b": 20,
                    "c": 30,
                },
                {
                    "_target_": "tests.tron.utils.instantiate.BClass",
                    "a": 50,
                    "b": 60,
                    "c": 70,
                },
            ],
            {},
            [partial(AClass, b=20, c=30), BClass(a=50, b=60, c=70)],
            id="list_of_partial_class+missing",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.AClass", "b": 20, "c": 30},
            {"a": 10, "d": 40},
            AClass(10, 20, 30, 40),
            id="class+override",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.AClass", "b": 20, "c": 30},
            {"a": 10, "_partial_": True},
            partial(AClass, a=10, b=20, c=30),
            id="class+override+partial1",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.AClass", "b": 20, "c": 30},
            {"a": "???", "_partial_": True},
            partial(AClass, b=20, c=30),
            id="class+override+partial1+missing",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.AClass",
                "_partial_": True,
                "c": 30,
            },
            {"a": 10, "d": 40},
            partial(AClass, a=10, c=30, d=40),
            id="class+override+partial2",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.AClass", "b": 200, "c": "${b}"},
            {"a": 10, "b": 99, "d": 40},
            AClass(10, 99, 99, 40),
            id="class+override+interpolation",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.AClass", "b": 200, "c": "${b}"},
            {"a": 10, "b": 99, "_partial_": True},
            partial(AClass, a=10, b=99, c=99),
            id="class+override+interpolation+partial1",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.AClass",
                "b": 200,
                "_partial_": True,
                "c": "${b}",
            },
            {"a": 10, "b": 99},
            partial(AClass, a=10, b=99, c=99),
            id="class+override+interpolation+partial2",
        ),
        # Check class and static methods
        param(
            {
                "_target_": "tests.tron.utils.instantiate.ASubclass.class_method",
                "_partial_": True,
            },
            {},
            partial(ASubclass.class_method),
            id="class_method+partial",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.ASubclass.class_method",
                "y": 10,
            },
            {},
            ASubclass(11),
            id="class_method",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.AClass.static_method",
                "_partial_": True,
            },
            {},
            partial(AClass.static_method),
            id="static_method+partial",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.AClass.static_method",
                "_partial_": True,
                "y": "???",
            },
            {},
            partial(AClass.static_method),
            id="static_method+partial+missing",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.AClass.static_method", "z": 43},
            {},
            43,
            id="static_method",
        ),
        # Check nested types and static methods
        param(
            {"_target_": "tests.tron.utils.instantiate.NestingClass"},
            {},
            NestingClass(ASubclass(10)),
            id="class_with_nested_class",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.nesting.a.class_method",
                "_partial_": True,
            },
            {},
            partial(ASubclass.class_method),
            id="class_method_on_an_object_nested_in_a_global+partial",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.nesting.a.class_method",
                "y": 10,
            },
            {},
            ASubclass(11),
            id="class_method_on_an_object_nested_in_a_global",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.nesting.a.static_method",
                "_partial_": True,
            },
            {},
            partial(ASubclass.static_method),
            id="static_method_on_an_object_nested_in_a_global+partial",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.nesting.a.static_method",
                "z": 43,
            },
            {},
            43,
            id="static_method_on_an_object_nested_in_a_global",
        ),
        # Check that default value is respected
        param(
            {"_target_": "tests.tron.utils.instantiate.AClass"},
            {"a": 10, "b": 20, "_partial_": True, "d": "new_default"},
            partial(AClass, a=10, b=20, d="new_default"),
            id="instantiate_respects_default_value+partial",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.AClass"},
            {"a": 10, "b": 20, "c": 30},
            AClass(10, 20, 30, "default_value"),
            id="instantiate_respects_default_value",
        ),
        # call a function from a module
        param(
            {
                "_target_": "tests.tron.utils.instantiate.module_function",
                "_partial_": True,
            },
            {},
            partial(module_function),
            id="call_function_in_module",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.module_function", "x": 43},
            {},
            43,
            id="call_function_in_module",
        ),
        # Check builtins
        param(
            {"_target_": "builtins.int", "base": 2, "_partial_": True},
            {},
            partial(int, base=2),
            id="builtin_types+partial",
        ),
        param(
            {"_target_": "builtins.str", "object": 43},
            {},
            "43",
            id="builtin_types",
        ),
        # passthrough
        param(
            {"_target_": "tests.tron.utils.instantiate.AClass"},
            {"a": 10, "b": 20, "c": 30},
            AClass(a=10, b=20, c=30),
            id="passthrough",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.AClass"},
            {"a": 10, "b": 20, "_partial_": True},
            partial(AClass, a=10, b=20),
            id="passthrough+partial",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.AClass"},
            {"a": 10, "b": 20, "c": 30, "d": {"x": IllegalType()}},
            AClass(a=10, b=20, c=30, d={"x": IllegalType()}),
            id="oc_incompatible_passthrough",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.AClass", "_partial_": True},
            {"a": 10, "b": 20, "d": {"x": IllegalType()}},
            partial(AClass, a=10, b=20, d={"x": IllegalType()}),
            id="oc_incompatible_passthrough+partial",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.AClass", "_partial_": True},
            {
                "a": 10,
                "b": 20,
                "d": {"x": [10, IllegalType()]},
            },
            partial(AClass, a=10, b=20, d={"x": [10, IllegalType()]}),
            id="passthrough:list+partial",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.AClass"},
            {
                "a": 10,
                "b": 20,
                "c": 30,
                "d": {"x": [10, IllegalType()]},
            },
            AClass(a=10, b=20, c=30, d={"x": [10, IllegalType()]}),
            id="passthrough:list",
        ),
        param(
            UntypedPassthroughConf,
            {"a": IllegalType()},
            UntypedPassthroughClass(a=IllegalType()),
            id="untyped_passthrough",
        ),
        param(
            KeywordsInParamsClass,
            {
                "_target_": "tests.tron.utils.instantiate.KeywordsInParamsClass",
                "target": "foo",
                "partial": "bar",
            },
            KeywordsInParamsClass(target="foo", partial="bar"),
            id="keywords_in_params",
        ),
        param([], {}, [], id="list_as_toplevel0"),
        param(
            [
                {
                    "_target_": "tests.tron.utils.instantiate.AClass",
                    "a": 10,
                    "b": 20,
                    "c": 30,
                    "d": 40,
                },
                {
                    "_target_": "tests.tron.utils.instantiate.BClass",
                    "a": 50,
                    "b": 60,
                    "c": 70,
                    "d": 80,
                },
            ],
            {},
            [AClass(10, 20, 30, 40), BClass(50, 60, 70, 80)],
            id="list_as_toplevel2",
        ),
    ],
)
def test_class_instantiate(
    config: Any,
    passthrough: Dict[str, Any],
    expected: Any,
) -> Any:
    original_config_str = str(config)
    obj = instantiate(config, **passthrough)
    assert partial_equal(obj, expected)
    assert str(config) == original_config_str


def test_partial_with_missing() -> Any:
    config = {
        "_target_": "tests.tron.utils.instantiate.AClass",
        "_partial_": True,
        "a": "???",
        "b": 20,
        "c": 30,
    }
    original_config_str = str(config)
    partial_obj = instantiate(config)
    assert partial_equal(partial_obj, partial(AClass, b=20, c=30))
    obj = partial_obj(a=10)
    assert partial_equal(obj, AClass(a=10, b=20, c=30))
    assert str(config) == original_config_str


def test_instantiate_with_missing() -> Any:
    config = {
        "_target_": "tests.tron.utils.instantiate.AClass",
        "a": "???",
        "b": 20,
        "c": 30,
    }
    with raises(MissingMandatoryValue, match=re.escape("Missing mandatory value: a")):
        instantiate(config)


def test_none_cases() -> Any:
    assert instantiate(None) is None

    cfg = {
        "_target_": "tests.tron.utils.instantiate.ArgsClass",
        "none_dict": DictConfig(None),
        "none_list": ListConfig(None),
        "dict": {
            "field": 10,
            "none_dict": DictConfig(None),
            "none_list": ListConfig(None),
        },
        "list": [
            10,
            DictConfig(None),
            ListConfig(None),
        ],
    }
    original_config_str = str(cfg)
    ret = instantiate(cfg)
    assert ret.kwargs["none_dict"] is None
    assert ret.kwargs["none_list"] is None
    assert ret.kwargs["dict"]["field"] == 10
    assert ret.kwargs["dict"]["none_dict"] is None
    assert ret.kwargs["dict"]["none_list"] is None
    assert ret.kwargs["list"][0] == 10
    assert ret.kwargs["list"][1] is None
    assert ret.kwargs["list"][2] is None
    assert str(cfg) == original_config_str


@mark.parametrize("convert_to_list", [True, False])
@mark.parametrize(
    "input_conf, passthrough, expected",
    [
        param(
            {
                "node": {
                    "_target_": "tests.tron.utils.instantiate.AClass",
                    "a": "${value}",
                    "b": 20,
                    "c": 30,
                    "d": 40,
                },
                "value": 99,
            },
            {},
            AClass(99, 20, 30, 40),
            id="interpolation_into_parent",
        ),
        param(
            {
                "node": {
                    "_target_": "tests.tron.utils.instantiate.AClass",
                    "_partial_": True,
                    "a": "${value}",
                    "b": 20,
                },
                "value": 99,
            },
            {},
            partial(AClass, a=99, b=20),
            id="interpolation_into_parent_partial",
        ),
        param(
            {
                "A": {
                    "_target_": "tests.tron.utils.instantiate.add_values",
                    "a": 1,
                    "b": 2,
                },
                "node": {
                    "_target_": "tests.tron.utils.instantiate.add_values",
                    "_partial_": True,
                    "a": "${A}",
                },
            },
            {},
            partial(add_values, a=3),
            id="interpolation_from_recursive_partial",
        ),
        param(
            {
                "A": {
                    "_target_": "tests.tron.utils.instantiate.add_values",
                    "a": 1,
                    "b": 2,
                },
                "node": {
                    "_target_": "tests.tron.utils.instantiate.add_values",
                    "a": "${A}",
                    "b": 3,
                },
            },
            {},
            6,
            id="interpolation_from_recursive",
        ),
        param(
            {
                "my_id": 5,
                "node": {
                    "b": "${foo_b}",
                },
                "foo_b": {
                    "unique_id": "${my_id}",
                },
            },
            {},
            OmegaConf.create({"b": {"unique_id": 5}}),
            id="interpolation_from_parent_with_interpolation",
        ),
        param(
            {
                "my_id": 5,
                "node": "${foo_b}",
                "foo_b": {
                    "unique_id": "${my_id}",
                },
            },
            {},
            OmegaConf.create({"unique_id": 5}),
            id="interpolation_from_parent_with_interpolation",
        ),
        param(
            DictConfig(
                {
                    "username": "test_user",
                    "node": {
                        "_target_": "tests.tron.utils.instantiate.TargetWithInstantiateInInit",
                        "user_config": {
                            "name": "${foo_b.username}",
                            "age": 40,
                        },
                    },
                    "foo_b": {
                        "username": "${username}",
                    },
                }
            ),
            {},
            TargetWithInstantiateInInit(user_config=None, user=User(name="test_user", age=40)),
            id="target_with_instantiate_in_init",
        ),
    ],
)
def test_interpolation_accessing_parent(
    input_conf: Any,
    passthrough: Dict[str, Any],
    expected: Any,
    convert_to_list: bool,
) -> Any:
    if convert_to_list:
        input_conf = copy.deepcopy(input_conf)
        input_conf["node"] = [input_conf["node"]]
    cfg_copy = OmegaConf.create(input_conf)
    input_conf = OmegaConf.create(input_conf)
    if convert_to_list:
        obj = instantiate(
            input_conf.node[0],
            **passthrough,
        )
    else:
        obj = instantiate(
            input_conf.node,
            **passthrough,
        )
    if isinstance(expected, partial):
        assert partial_equal(obj, expected)
    else:
        assert obj == expected
    assert input_conf == cfg_copy


@mark.parametrize(
    "src",
    [
        (
            {
                "_target_": "tests.tron.utils.instantiate.AClass",
                "b": 200,
                "c": {"x": 10, "y": "${b}"},
            }
        )
    ],
)
def test_class_instantiate_omegaconf_node(config: Any) -> Any:
    obj = instantiate(
        config,
        a=10,
        d={"_target_": "tests.tron.utils.instantiate.AnotherClass", "x": 99},
    )
    assert obj == AClass(a=10, b=200, c={"x": 10, "y": 200}, d=AnotherClass(99))


@mark.parametrize(
    "src",
    [
        (
            ListConfig(
                [
                    {
                        "_target_": "tests.tron.utils.instantiate.AClass",
                        "b": 200,
                        "c": {"x": 10, "y": "${0.b}"},
                    }
                ]
            )
        )
    ],
)
def test_class_instantiate_list_item(config: Any) -> Any:
    obj = instantiate(
        config[0],
        a=10,
        d={"_target_": "tests.tron.utils.instantiate.AnotherClass", "x": 99},
    )
    assert obj == AClass(a=10, b=200, c={"x": 10, "y": 200}, d=AnotherClass(99))


@mark.parametrize("src", [{"_target_": "tests.tron.utils.instantiate.Adam"}])
def test_instantiate_adam(config: Any) -> None:
    with raises(
        InstantiationException,
        match=r"Error in call to target 'tests\.tron\.utils\.instantiate\.Adam':\nTypeError\(.*\)",
    ):
        # can't instantiate without passing params
        instantiate(config)

    adam_params = Parameters([1, 2, 3])
    res = instantiate(config, params=adam_params)
    assert res == Adam(params=adam_params)


@mark.parametrize("is_partial", [True, False])
def test_regression_1483(is_partial: bool) -> None:
    """
    In 1483, pickle is failing because the parent node of lst node contains
    a generator, which is not picklable.
    The solution is to resolve and retach from the parent before calling the function.
    This tests verifies the expected behavior.
    """

    def gen() -> Any:
        yield 10

    res: ArgsClass = instantiate(
        {"_target_": "tests.tron.utils.instantiate.ArgsClass"},
        _partial_=is_partial,
        gen=gen(),
        lst=[1, 2],
    )
    if is_partial:
        # res is of type functools.partial
        pickle.dumps(res.keywords["lst"])  # type: ignore
    else:
        pickle.dumps(res.kwargs["lst"])


@mark.parametrize(
    "is_partial,expected_params",
    [(True, Parameters([1, 2, 3])), (False, partial(Parameters))],
)
def test_instantiate_adam_conf(is_partial: bool, expected_params: Any) -> None:
    with raises(
        InstantiationException,
        match=r"Error in call to target 'tests\.tron\.utils\.instantiate\.Adam':\nTypeError\(.*\)",
    ):
        # can't instantiate without passing params
        instantiate(AdamConf())

    adam_params = expected_params
    res = instantiate(AdamConf(lr=0.123), params=adam_params)
    expected = Adam(lr=0.123, params=adam_params)
    if is_partial:
        partial_equal(res.params, expected.params)
    else:
        assert res.params == expected.params
    assert res.lr == expected.lr
    assert list(res.betas) == list(expected.betas)  # OmegaConf converts tuples to lists
    assert res.eps == expected.eps
    assert res.weight_decay == expected.weight_decay
    assert res.amsgrad == expected.amsgrad


def test_instantiate_adam_conf_with_convert() -> None:
    adam_params = Parameters([1, 2, 3])
    res = instantiate(
        AdamConf(lr=0.123),
        params=adam_params,
    )
    expected = Adam(lr=0.123, params=adam_params)
    assert res.params == expected.params
    assert res.lr == expected.lr
    assert isinstance(res.betas, list)
    assert list(res.betas) == list(expected.betas)  # OmegaConf converts tuples to lists
    assert res.eps == expected.eps
    assert res.weight_decay == expected.weight_decay
    assert res.amsgrad == expected.amsgrad


def test_instantiate_with_missing_module() -> None:
    _target_ = "tests.tron.utils.instantiate.ClassWithMissingModule"
    with raises(
        InstantiationException,
        match=dedent(
            rf"""
            Error in call to target '{re.escape(_target_)}':
            ModuleNotFoundError\("No module named 'some_missing_module'",?\)"""
        ).strip(),
    ):
        # can't instantiate when importing a missing module
        instantiate({"_target_": _target_})


def test_instantiate_target_raising_exception_taking_no_arguments() -> None:
    _target_ = "tests.tron.utils.instantiate.raise_exception_taking_no_argument"
    with raises(
        InstantiationException,
        match=(
            dedent(
                rf"""
                Error in call to target '{re.escape(_target_)}':
                ExceptionTakingNoArgument\('Err message',?\)"""
            ).strip()
        ),
    ):
        instantiate({}, _target_=_target_)


def test_instantiate_target_raising_exception_taking_no_arguments_nested() -> None:
    _target_ = "tests.tron.utils.instantiate.raise_exception_taking_no_argument"
    with raises(
        InstantiationException,
        match=(
            dedent(
                rf"""
                Error in call to target '{re.escape(_target_)}':
                ExceptionTakingNoArgument\('Err message',?\)
                full_key: foo
                """
            ).strip()
        ),
    ):
        instantiate({"foo": {"_target_": _target_}})


def test_toplevel_list_partial_not_allowed() -> None:
    config = [{"_target_": "tests.tron.utils.instantiate.ClassA", "a": 10, "b": 20, "c": 30}]
    with raises(
        InstantiationException,
        match=re.escape("The _partial_ keyword is not compatible with top-level list instantiation"),
    ):
        instantiate(config, _partial_=True)


@mark.parametrize("is_partial", [True, False])
def test_pass_extra_variables(is_partial: bool) -> None:
    cfg = OmegaConf.create(
        {
            "_target_": "tests.tron.utils.instantiate.AClass",
            "a": 10,
            "b": 20,
            "_partial_": is_partial,
        }
    )
    if is_partial:
        assert partial_equal(instantiate(cfg, c=30), partial(AClass, a=10, b=20, c=30))
    else:
        assert instantiate(cfg, c=30) == AClass(a=10, b=20, c=30)


@mark.parametrize(
    "target, expected",
    [
        param(module_function2, lambda x: x == "fn return", id="fn"),
        param(OuterClass, lambda x: isinstance(x, OuterClass), id="OuterClass"),
        param(
            OuterClass.method,
            lambda x: x == "OuterClass.method return",
            id="classmethod",
        ),
        param(OuterClass.Nested, lambda x: isinstance(x, OuterClass.Nested), id="nested"),
        param(
            OuterClass.Nested.method,
            lambda x: x == "OuterClass.Nested.method return",
            id="nested_method",
        ),
    ],
)
def test_instantiate_with_callable_target_keyword(target: Callable[[], None], expected: Callable[[Any], bool]) -> None:
    ret = instantiate({}, _target_=target)
    assert expected(ret)


@mark.parametrize(
    "src, passthrough, expected",
    [
        # direct
        param(
            {
                "_target_": "tests.tron.utils.instantiate.Tree",
                "value": 1,
                "left": {
                    "_target_": "tests.tron.utils.instantiate.Tree",
                    "value": 21,
                },
                "right": {
                    "_target_": "tests.tron.utils.instantiate.Tree",
                    "value": 22,
                },
            },
            {},
            Tree(value=1, left=Tree(value=21), right=Tree(value=22)),
            id="recursive:direct:dict",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.Tree"},
            {"value": 1},
            Tree(value=1),
            id="recursive:direct:dict:passthrough",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.Tree"},
            {
                "value": 1,
                "left": {"_target_": "tests.tron.utils.instantiate.Tree", "value": 2},
            },
            Tree(value=1, left=Tree(2)),
            id="recursive:direct:dict:passthrough",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.Tree"},
            {
                "value": 1,
                "left": {"_target_": "tests.tron.utils.instantiate.Tree", "value": 2},
                "right": {"_target_": "tests.tron.utils.instantiate.Tree", "value": 3},
            },
            Tree(value=1, left=Tree(2), right=Tree(3)),
            id="recursive:direct:dict:passthrough",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.Tree"},
            {"value": IllegalType()},
            Tree(value=IllegalType()),
            id="recursive:direct:dict:passthrough:incompatible_value",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.Tree"},
            {
                "value": 1,
                "left": {
                    "_target_": "tests.tron.utils.instantiate.Tree",
                    "value": IllegalType(),
                },
            },
            Tree(value=1, left=Tree(value=IllegalType())),
            id="recursive:direct:dict:passthrough:incompatible_value",
        ),
        param(
            TreeConf(
                value=1,
                left=TreeConf(value=21),
                right=TreeConf(value=22),
            ),
            {},
            Tree(value=1, left=Tree(value=21), right=Tree(value=22)),
            id="recursive:direct:dataclass",
        ),
        param(
            TreeConf(
                value=1,
                left=TreeConf(value=21),
            ),
            {"right": {"value": 22}},
            Tree(value=1, left=Tree(value=21), right=Tree(value=22)),
            id="recursive:direct:dataclass:passthrough",
        ),
        param(
            TreeConf(
                value=1,
                left=TreeConf(value=21),
            ),
            {"right": TreeConf(value=22)},
            Tree(value=1, left=Tree(value=21), right=Tree(value=22)),
            id="recursive:direct:dataclass:passthrough",
        ),
        param(
            TreeConf(
                value=1,
                left=TreeConf(value=21),
            ),
            {
                "right": TreeConf(value=IllegalType()),
            },
            Tree(value=1, left=Tree(value=21), right=Tree(value=IllegalType())),
            id="recursive:direct:dataclass:passthrough",
        ),
        # list
        # note that passthrough to a list element is not currently supported
        param(
            ComposeConf(
                transforms=[
                    CenterCropConf(size=10),
                    RotationConf(degrees=45),
                ]
            ),
            {},
            Compose(
                transforms=[
                    CenterCrop(size=10),
                    Rotation(degrees=45),
                ]
            ),
            id="recursive:list:dataclass",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.Compose",
                "transforms": [
                    {
                        "_target_": "tests.tron.utils.instantiate.CenterCrop",
                        "size": 10,
                    },
                    {
                        "_target_": "tests.tron.utils.instantiate.Rotation",
                        "degrees": 45,
                    },
                ],
            },
            {},
            Compose(
                transforms=[
                    CenterCrop(size=10),
                    Rotation(degrees=45),
                ]
            ),
            id="recursive:list:dict",
        ),
        # map
        param(
            MappingConf(
                dictionary={
                    "a": MappingConf(),
                    "b": MappingConf(),
                }
            ),
            {},
            Mapping(
                dictionary={
                    "a": Mapping(),
                    "b": Mapping(),
                }
            ),
            id="recursive:map:dataclass",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.Mapping",
                "dictionary": {
                    "a": {"_target_": "tests.tron.utils.instantiate.Mapping"},
                    "b": {"_target_": "tests.tron.utils.instantiate.Mapping"},
                },
            },
            {},
            Mapping(
                dictionary={
                    "a": Mapping(),
                    "b": Mapping(),
                }
            ),
            id="recursive:map:dict",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.Mapping",
                "dictionary": {
                    "a": {"_target_": "tests.tron.utils.instantiate.Mapping"},
                },
            },
            {
                "dictionary": {
                    "b": {"_target_": "tests.tron.utils.instantiate.Mapping"},
                },
            },
            Mapping(
                dictionary={
                    "a": Mapping(),
                    "b": Mapping(),
                }
            ),
            id="recursive:map:dict:passthrough",
        ),
    ],
)
def test_recursive_instantiation(
    config: Any,
    passthrough: Dict[str, Any],
    expected: Any,
) -> None:
    obj = instantiate(config, **passthrough)
    assert obj == expected


@mark.parametrize(
    "src, passthrough, expected",
    [
        # direct
        param(
            {
                "_target_": "tests.tron.utils.instantiate.Tree",
                "_partial_": True,
                "left": {
                    "_target_": "tests.tron.utils.instantiate.Tree",
                    "value": 21,
                },
                "right": {
                    "_target_": "tests.tron.utils.instantiate.Tree",
                    "value": 22,
                },
            },
            {},
            partial(Tree, left=Tree(value=21), right=Tree(value=22)),
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.Tree", "_partial_": True},
            {"value": 1},
            partial(Tree, value=1),
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.Tree"},
            {
                "value": 1,
                "left": {
                    "_target_": "tests.tron.utils.instantiate.Tree",
                    "_partial_": True,
                },
            },
            Tree(value=1, left=partial(Tree)),
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.Tree"},
            {
                "value": 1,
                "left": {
                    "_target_": "tests.tron.utils.instantiate.Tree",
                    "_partial_": True,
                },
                "right": {"_target_": "tests.tron.utils.instantiate.Tree", "value": 3},
            },
            Tree(value=1, left=partial(Tree), right=Tree(3)),
        ),
        param(
            TreeConf(
                value=1,
                left=TreeConf(value=21, _partial_=True),
                right=TreeConf(value=22),
            ),
            {},
            Tree(
                value=1,
                left=partial(Tree, value=21, left=None, right=None),
                right=Tree(value=22),
            ),
        ),
        param(
            TreeConf(
                _partial_=True,
                value=1,
                left=TreeConf(value=21, _partial_=True),
                right=TreeConf(value=22, _partial_=True),
            ),
            {},
            partial(
                Tree,
                value=1,
                left=partial(Tree, value=21, left=None, right=None),
                right=partial(Tree, value=22, left=None, right=None),
            ),
        ),
        param(
            TreeConf(
                _partial_=True,
                value=1,
                left=TreeConf(
                    value=21,
                ),
                right=TreeConf(value=22, left=TreeConf(_partial_=True, value=42)),
            ),
            {},
            partial(
                Tree,
                value=1,
                left=Tree(value=21),
                right=Tree(value=22, left=partial(Tree, value=42, left=None, right=None)),
            ),
        ),
        # list
        # note that passthrough to a list element is not currently supported
        param(
            ComposeConf(
                _partial_=True,
                transforms=[
                    CenterCropConf(size=10),
                    RotationConf(degrees=45),
                ],
            ),
            {},
            partial(
                Compose,
                transforms=[
                    CenterCrop(size=10),
                    Rotation(degrees=45),
                ],
            ),
        ),
        param(
            ComposeConf(
                transforms=[
                    CenterCropConf(_partial_=True, size=10),
                    RotationConf(degrees=45),
                ],
            ),
            {},
            Compose(
                transforms=[
                    partial(CenterCrop, size=10),  # type: ignore
                    Rotation(degrees=45),
                ],
            ),
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.Compose",
                "transforms": [
                    {
                        "_target_": "tests.tron.utils.instantiate.CenterCrop",
                        "_partial_": True,
                    },
                    {
                        "_target_": "tests.tron.utils.instantiate.Rotation",
                        "degrees": 45,
                    },
                ],
            },
            {},
            Compose(
                transforms=[
                    partial(CenterCrop),  # type: ignore
                    Rotation(degrees=45),
                ]
            ),
            id="recursive:list:dict",
        ),
        # map
        param(
            MappingConf(
                dictionary={
                    "a": MappingConf(_partial_=True),
                    "b": MappingConf(),
                }
            ),
            {},
            Mapping(
                dictionary={
                    "a": partial(Mapping, dictionary=None),  # type: ignore
                    "b": Mapping(),
                }
            ),
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.Mapping",
                "_partial_": True,
                "dictionary": {
                    "a": {
                        "_target_": "tests.tron.utils.instantiate.Mapping",
                        "_partial_": True,
                    },
                },
            },
            {
                "dictionary": {
                    "b": {
                        "_target_": "tests.tron.utils.instantiate.Mapping",
                        "_partial_": True,
                    },
                },
            },
            partial(
                Mapping,
                dictionary={
                    "a": partial(Mapping),
                    "b": partial(Mapping),
                },
            ),
        ),
    ],
)
def test_partial_instantiate(
    config: Any,
    passthrough: Dict[str, Any],
    expected: Any,
) -> None:
    obj = instantiate(config, **passthrough)
    assert obj == expected or partial_equal(obj, expected)


@mark.parametrize(
    ("src", "passthrough", "expected"),
    [
        param(
            {
                "_target_": "tests.tron.utils.instantiate.Tree",
                "value": 1,
                "left": {
                    "_target_": "tests.tron.utils.instantiate.Tree",
                    "value": 21,
                },
            },
            {},
            Tree(value=1, left=Tree(value=21)),
            id="default",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.Tree",
                "value": 1,
                "left": {
                    "_target_": "tests.tron.utils.instantiate.Tree",
                    "value": 2,
                    "left": {
                        "_target_": "tests.tron.utils.instantiate.Tree",
                        "value": 3,
                    },
                },
            },
            {},
            Tree(value=1, left=Tree(value=2, left=Tree(value=3))),
            id="3_levels:default",
        ),
    ],
)
def test_recursive_override(
    config: Any,
    passthrough: Any,
    expected: Any,
) -> None:
    obj = instantiate(config, **passthrough)
    assert obj == expected


@mark.parametrize(
    ("src", "passthrough", "expected"),
    [
        param(
            {
                "_target_": "tests.tron.utils.instantiate.AClass",
                "a": 10,
                "b": 20,
                "c": 30,
                "d": 40,
            },
            {"_target_": "tests.tron.utils.instantiate.BClass"},
            BClass(10, 20, 30, 40),
            id="str:override_same_args",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.AClass",
                "a": 10,
                "b": 20,
                "c": 30,
                "d": 40,
            },
            {"_target_": BClass},
            BClass(10, 20, 30, 40),
            id="type:override_same_args",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.AClass", "a": 10, "b": 20},
            {"_target_": "tests.tron.utils.instantiate.BClass"},
            BClass(10, 20, "c", "d"),
            id="str:override_other_args",
        ),
        param(
            {"_target_": "tests.tron.utils.instantiate.AClass", "a": 10, "b": 20},
            {"_target_": BClass},
            BClass(10, 20, "c", "d"),
            id="type:override_other_args",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.AClass",
                "a": 10,
                "b": 20,
                "c": {
                    "_target_": "tests.tron.utils.instantiate.AClass",
                    "a": "aa",
                    "b": "bb",
                    "c": "cc",
                },
            },
            {
                "_target_": "tests.tron.utils.instantiate.BClass",
                "c": {
                    "_target_": "tests.tron.utils.instantiate.BClass",
                },
            },
            BClass(10, 20, BClass(a="aa", b="bb", c="cc"), "d"),
            id="str:recursive_override",
        ),
        param(
            {
                "_target_": "tests.tron.utils.instantiate.AClass",
                "a": 10,
                "b": 20,
                "c": {
                    "_target_": "tests.tron.utils.instantiate.AClass",
                    "a": "aa",
                    "b": "bb",
                    "c": "cc",
                },
            },
            {"_target_": BClass, "c": {"_target_": BClass}},
            BClass(10, 20, BClass(a="aa", b="bb", c="cc"), "d"),
            id="type:recursive_override",
        ),
    ],
)
def test_override_target(config: Any, passthrough: Any, expected: Any) -> None:
    obj = instantiate(config, **passthrough)
    assert obj == expected


@mark.parametrize(
    "config, passthrough, expected",
    [
        param(
            {"_target_": AnotherClass, "x": 10},
            {},
            AnotherClass(10),
            id="class_in_config_dict",
        ),
    ],
)
def test_instantiate_from_class_in_dict(config: Any, passthrough: Any, expected: Any) -> None:
    config_copy = copy.deepcopy(config)
    assert instantiate(config, **passthrough) == expected
    assert config == config_copy


@mark.parametrize(
    "config, passthrough, err_msg",
    [
        param(
            OmegaConf.create({"_target_": AClass}),
            {},
            re.escape(
                "Expected a callable target, got"
                + " '{'a': '???', 'b': '???', 'c': '???', 'd': 'default_value'}' of type 'DictConfig'"
            ),
            id="instantiate-from-dataclass-in-dict-fails",
        ),
        param(
            OmegaConf.create({"foo": {"_target_": AClass}}),
            {},
            re.escape(
                "Expected a callable target, got"
                + " '{'a': '???', 'b': '???', 'c': '???', 'd': 'default_value'}' of type 'DictConfig'"
                + "\nfull_key: foo"
            ),
            id="instantiate-from-dataclass-in-dict-fails-nested",
        ),
    ],
)
def test_instantiate_from_dataclass_in_dict_fails(config: Any, passthrough: Any, err_msg: str) -> None:
    with raises(
        InstantiationException,
        match=err_msg,
    ):
        instantiate(config, **passthrough)


def test_cannot_locate_target() -> None:
    cfg = OmegaConf.create({"foo": {"_target_": "not_found"}})
    with raises(
        InstantiationException,
        match=re.escape(dedent("Error locating target 'not_found'")),
    ):
        instantiate(cfg)


def test_locate_empty_path():
    """Test _locate with empty path (line 42)"""
    with raises(ImportError, match="Empty path"):
        _locate("")


def test_locate_invalid_dotstring():
    """Test _locate with invalid dotstring (line 49)"""
    with raises(ValueError, match="Error loading '.invalid.': invalid dotstring"):
        _locate(".invalid.")


def test_locate_attribute_error():
    """Test _locate with attribute error (lines 63-64)"""
    with raises(ImportError, match=re.escape("Error loading 'os.nonexistent_attribute':")):
        _locate("os.nonexistent_attribute")


def test_locate_general_exception():
    """Test _locate handling general exceptions (lines 75-77)"""
    # Creating a test file that will raise an exception when imported
    test_module_path = os.path.join(os.path.dirname(__file__), "test_error_module.py")

    try:
        # Create a module that raises an exception when imported
        with open(test_module_path, "w") as f:
            f.write("raise RuntimeError('Test exception when importing')\n")

        # Try to import it
        with raises(
            ImportError,
            match="Error loading 'tests.tron.utils.instantiate.test_error_module'",
        ):
            _locate("tests.tron.utils.instantiate.test_error_module")
    finally:
        # Clean up the test file
        if os.path.exists(test_module_path):
            os.remove(test_module_path)


def test_locate_module_not_found():
    """Test _locate with module not found error (line 88-91)"""
    with raises(ImportError, match="Unable to import any module in the path"):
        _locate("nonexistent_module.some_function")


def test_convert_target_to_string_non_callable():
    """Test _convert_target_to_string with non-callable object (lines 106-110)"""
    # Test with non-callable object
    obj = 42
    result = _convert_target_to_string(obj)
    assert result == 42


def test_instantiate_unsupported_type():
    """Test instantiate with unsupported type (lines 249-261)"""
    with raises(InstantiationException, match="Cannot instantiate config of type"):
        instantiate(42)


def test_list_instantiation():
    """Test list instantiation functionality (lines 288-291)"""
    config = ListConfig([1, 2, {"_target_": "builtins.str", "object": 3}])
    result = instantiate_node(config)
    assert result == [1, 2, "3"]


# Create a simple class for testing instantiate_node functions
class SimpleCoverageClass:
    def __init__(self, a: int = 10, b: str = "default"):
        self.a = a
        self.b = b


def test_instantiate_node_error_with_call_false_and_extra_keys():
    """Test error handling in instantiate_node with _call_=False and extra keys (lines 306-312)"""
    config = DictConfig(
        {
            "_target_": "tests.tron.utils.instantiate.test_instantiate.SimpleCoverageClass",
            "_call_": False,
            "a": 10,  # Extra key that should cause an error
        }
    )

    with raises(InstantiationException, match="but extra keys were found"):
        instantiate_node(config)


def test_instantiate_node_with_call_false():
    """Test instantiate_node with _call_=False (lines 312)"""
    config = DictConfig(
        {
            "_target_": "utils.instantiate.test_instantiate.SimpleCoverageClass",
            "_call_": False,
        }
    )

    result = instantiate_node(config)
    assert result == SimpleCoverageClass


def test_instantiate_node_error_lenient_mode(caplog):
    """Test error handling in instantiate_node in lenient mode (lines 321-326)"""
    caplog.set_level(logging.WARNING)

    # Create a config with a value that will fail to instantiate
    config = DictConfig(
        {
            "_target_": "utils.instantiate.test_instantiate.SimpleCoverageClass",
            "a": {"_target_": "nonexistent.module.Class"},  # This will fail to resolve
        }
    )

    # Instantiate in lenient mode
    result = instantiate_node(config, mode=InstantiationMode.LENIENT)

    # Check that warning was logged
    assert "Error instantiating" in caplog.text
    assert "Using None instead in lenient mode" in caplog.text

    # Check that the object was created with a=None
    assert isinstance(result, SimpleCoverageClass)
    assert result.a is None


def test_unexpected_config_type():
    """Test handling of unexpected config types (line 340)"""

    # Create a mock object that is OmegaConf.is_config but not a dict or list
    class MockConfig:
        def _is_none(self):
            return False

        # Add other required methods that might be called
        def __getitem__(self, key):
            return None

        def keys(self):
            return []

        def _iter_ex(self, resolve):
            return []

        def _get_full_key(self, key):
            return "mock"

    mock_config = MockConfig()

    # Patch the OmegaConf.is_config and is_list/is_dict methods to make our mock object appear as a config
    original_is_config = OmegaConf.is_config
    original_is_list = OmegaConf.is_list
    original_is_dict = OmegaConf.is_dict

    try:
        OmegaConf.is_config = lambda x: isinstance(x, MockConfig) or original_is_config(x)
        OmegaConf.is_list = lambda x: not isinstance(x, MockConfig) and original_is_list(x)
        OmegaConf.is_dict = lambda x: not isinstance(x, MockConfig) and original_is_dict(x)

        with raises(AssertionError, match="Unexpected config type"):
            instantiate_node(mock_config)
    finally:
        # Restore original methods
        OmegaConf.is_config = original_is_config
        OmegaConf.is_list = original_is_list
        OmegaConf.is_dict = original_is_dict


def test_convert_node():
    """Test the _convert_node function"""
    # Test with OmegaConf config object
    config = OmegaConf.create({"a": 1, "b": 2})
    result = _convert_node(config)
    assert result == {"a": 1, "b": 2}

    # Test with regular object
    obj = SimpleCoverageClass()
    result = _convert_node(obj)
    assert result is obj  # Should return the object unchanged


def test_convert_target_to_string_callable():
    """Test _convert_target_to_string with callable object (lines 106-110)"""

    def sample_function():
        pass

    # Test with a function (callable)
    result = _convert_target_to_string(sample_function)
    assert result == f"{sample_function.__module__}.{sample_function.__qualname__}"

    # Test with a class (also callable)
    result = _convert_target_to_string(SimpleCoverageClass)
    assert result == f"{SimpleCoverageClass.__module__}.{SimpleCoverageClass.__qualname__}"


def test_resolve_target_full_key():
    """Test _resolve_target with full_key handling (line 145)"""
    from nemo.tron.utils.instantiate_utils import _resolve_target

    # Create a test case that will trigger the error and include full_key
    with raises(InstantiationException) as excinfo:
        _resolve_target("nonexistent.path", "test_full_key")

    # Verify the error message contains the full_key
    assert "full_key: test_full_key" in str(excinfo.value)


def test_locate_with_complex_path():
    """Additional test for _locate with more complex path (lines 88, 91)"""
    # Test a complex path with multiple components where part of it exists
    # but the whole path doesn't, to exercise more branches
    with raises(ImportError) as excinfo:
        _locate("os.path.nonexistent_submodule.nonexistent_function")

    error_message = str(excinfo.value)
    # Check for smaller parts that won't have escaping issues
    assert "os.path" in error_message
    assert "nonexistent_submodule" in error_message

    # Test a completely nonexistent module
    with raises(ImportError) as excinfo:
        _locate("completely_nonexistent_module.function")

    error_message = str(excinfo.value)
    assert "Unable to import any module in the path" in error_message
    assert "completely_nonexistent_module" in error_message


def test_instantiate_node_list_complex():
    """Test list instantiation with more complex structure (lines 288-291)"""
    # Create a list with nested dict configs and values
    config = ListConfig(
        [
            1,
            {"_target_": "builtins.str", "object": 3},
            {
                "_target_": "utils.instantiate.test_instantiate.SimpleCoverageClass",
                "a": 10,
                "b": "test",
            },
        ]
    )

    result = instantiate_node(config)

    # Check list elements were properly instantiated
    assert result[0] == 1
    assert result[1] == "3"
    assert isinstance(result[2], SimpleCoverageClass)
    assert result[2].a == 10
    assert result[2].b == "test"


def test_instantiate_node_error_strict_mode():
    """Test error handling in instantiate_node in strict mode (line 323)"""
    # Create a config with a value that will fail to instantiate
    config = DictConfig(
        {
            "_target_": "utils.instantiate.test_instantiate.SimpleCoverageClass",
            "a": {"_target_": "nonexistent.module.Class"},  # This will fail to resolve
        }
    )

    # In strict mode, this should raise an exception
    with raises(InstantiationException) as excinfo:
        instantiate_node(config, mode=InstantiationMode.STRICT)

    # Verify the error message
    assert "Error instantiating" in str(excinfo.value)


def test_partial_roundtrip():
    """Test serialization and deserialization of partial objects."""
    # Create partial objects
    simple_partial = partial(module_function)
    complex_partial = partial(module_function, arg=42, keyword="test")

    # Initialize test partials
    test_partials = [
        simple_partial,
        complex_partial,
        partial(AClass, a=10, b=20),
        partial(OuterClass.method),
        partial(AClass.static_method, y=5),
    ]

    for original_partial in test_partials:
        # Serialize to YAML
        with safe_yaml_representers():
            yaml_str = yaml.safe_dump(original_partial)
            yaml_dict = yaml.safe_load(yaml_str)

        # Deserialize using instantiate
        deserialized_partial = instantiate(yaml_dict)

        # Check that it's a partial
        assert isinstance(deserialized_partial, partial)

        # Check function identity - compare qualnames since the actual function objects might be different
        assert original_partial.func.__qualname__ == deserialized_partial.func.__qualname__

        # Check arguments if any
        if hasattr(original_partial, "args") and original_partial.args:
            assert len(original_partial.args) == len(deserialized_partial.args)
            for i, arg in enumerate(original_partial.args):
                if isinstance(arg, (int, float, str, bool, type(None))):
                    assert arg == deserialized_partial.args[i]

        # Check keywords if any
        if hasattr(original_partial, "keywords") and original_partial.keywords:
            assert len(original_partial.keywords) == len(deserialized_partial.keywords)
            for k, v in original_partial.keywords.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    assert k in deserialized_partial.keywords
                    assert v == deserialized_partial.keywords[k]


def test_partial_with_local_function_raises_error():
    """Test that partial objects with local functions fail to deserialize."""

    # Create a partial with simple parameters for reliable execution
    def add(a, b, c=0):
        return a + b + c

    original_partial = partial(add, 1, c=3)

    # Roundtrip through YAML
    with safe_yaml_representers():
        yaml_str = yaml.safe_dump(original_partial)
        yaml_dict = yaml.safe_load(yaml_str)

    # Deserialize should fail with an error because 'add' is a local function
    # that cannot be imported
    with raises(InstantiationException, match=re.compile("Error locating target.*", re.DOTALL)):
        instantiate(yaml_dict)


@mark.parametrize(
    "create_partial, execute_args, expected_result",
    [
        # Simple partial with one arg
        (lambda: partial(add_values, a=5), {"b": 3}, 8),
        # Partial with multiple args
        (lambda: partial(add_values, a=10, b=20), {}, 30),
        # Class constructor as partial
        (
            lambda: partial(SimpleClass, a=42),
            {"b": "hello"},
            SimpleClass(a=42, b="hello"),
        ),
    ],
)
def test_partial_complex_roundtrip(create_partial, execute_args, expected_result):
    """Test roundtrip with various types of partial objects and execution scenarios."""
    original_partial = create_partial()

    # Roundtrip through YAML
    with safe_yaml_representers():
        yaml_str = yaml.safe_dump(original_partial)
        yaml_dict = yaml.safe_load(yaml_str)

    # Deserialize
    deserialized_partial = instantiate(yaml_dict)

    # Test execution - should match expected_result
    original_result = original_partial(**execute_args)
    deserialized_result = deserialized_partial(**execute_args)

    # Compare results - handle different object types appropriately
    if isinstance(original_result, (list, tuple, dict)):
        assert original_result == deserialized_result
    elif hasattr(original_result, "__dict__"):
        # For objects, compare __dict__
        assert original_result.__dict__ == deserialized_result.__dict__
    else:
        assert original_result == deserialized_result


def test_torch_dtype_roundtrip():
    """Test serialization and deserialization of torch dtypes."""
    try:
        import torch

        # Test different torch dtypes
        dtypes_to_test = [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int32,
            torch.int64,
            torch.bool,
        ]

        for original_dtype in dtypes_to_test:
            # Serialize to YAML
            with safe_yaml_representers():
                yaml_str = yaml.safe_dump(original_dtype)
                yaml_dict = yaml.safe_load(yaml_str)

            # Verify the serialized structure
            assert "_target_" in yaml_dict
            assert yaml_dict["_target_"] == str(original_dtype)
            assert "_call_" in yaml_dict
            assert yaml_dict["_call_"] is False

            # Deserialize using instantiate
            deserialized_dtype = instantiate(yaml_dict)

            # Verify round-trip equality
            assert deserialized_dtype == original_dtype
            assert str(deserialized_dtype) == str(original_dtype)
            assert type(deserialized_dtype) is type(original_dtype)

        # Test in a nested structure
        nested_config = {
            "model_config": {
                "dtype": torch.bfloat16,
                "settings": {
                    "precision": torch.float32,
                    "inference_dtype": torch.float16,
                },
            }
        }

        # Serialize to YAML
        with safe_yaml_representers():
            yaml_str = yaml.safe_dump(nested_config)
            yaml_dict = yaml.safe_load(yaml_str)

        # Deserialize using instantiate
        deserialized_config = instantiate(yaml_dict)

        # Verify nested dtypes
        assert deserialized_config["model_config"]["dtype"] == torch.bfloat16
        assert deserialized_config["model_config"]["settings"]["precision"] == torch.float32
        assert deserialized_config["model_config"]["settings"]["inference_dtype"] == torch.float16

    except ImportError:
        pytest.skip("PyTorch not installed, skipping torch dtype tests")


def test_torch_dtype_from_yaml_file():
    """Test loading torch dtypes from a YAML file and instantiating the config."""
    try:
        import torch

        # Create another YAML file with multiple dtypes
        yaml_content = """
# List of dtypes
dtypes:
-   _target_: torch.float16
    _call_: false
-   _target_: torch.bfloat16
    _call_: false
-   _target_: torch.int8
    _call_: false
"""
        config_dict = yaml.safe_load(yaml_content)

        # Instantiate the config
        config = instantiate(config_dict)

        # Verify the dtypes in the list
        assert len(config["dtypes"]) == 3
        assert config["dtypes"][0] == torch.float16
        assert config["dtypes"][1] == torch.bfloat16
        assert config["dtypes"][2] == torch.int8

    except ImportError:
        pytest.skip("PyTorch not installed, skipping torch dtype tests")
