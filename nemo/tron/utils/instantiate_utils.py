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

# Patch for https://github.com/facebookresearch/hydra/blob/main/hydra/_internal/instantiate/_instantiate2.py
# until https://github.com/facebookresearch/hydra/issues/2140 is resolved

import copy
import functools
import logging
from enum import Enum
from textwrap import dedent
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import dataclasses

from omegaconf import OmegaConf
from omegaconf._utils import is_structured_config


class InstantiationException(Exception): ...


class InstantiationMode(Enum):
    """Enum for instantiation modes."""

    STRICT = "strict"
    LENIENT = "lenient"


class _Keys(str, Enum):
    """Special keys in configs used by instantiate."""

    TARGET = "_target_"
    PARTIAL = "_partial_"
    CALL = "_call_"
    ARGS = "_args_"


def instantiate(
    config: Any,
    *args: Any,
    mode: InstantiationMode = InstantiationMode.LENIENT,
    **kwargs: Any,
) -> Any:
    """
    :param config: An config object describing what to call and what params to use.
                   In addition to the parameters, the config must contain:
                   _target_ : target class or callable name (str)
                   And may contain:
                   _args_: List-like of positional arguments to pass to the target
                   _partial_: If True, return functools.partial wrapped method or object
                              False by default. Configure per target.
    :param args: Optional positional parameters pass-through
    :param kwargs: Optional named parameters to override
                   parameters in the config object. Parameters not present
                   in the config objects are being passed as is to the target.
                   IMPORTANT: dataclasses instances in kwargs are interpreted as config
                              and cannot be used as passthrough
    :return: if _target_ is a class name: the instantiated object
             if _target_ is a callable: the return value of the call
    """

    # Return None if config is None
    if config is None:
        return None

    if isinstance(config, (dict, list)):
        config = _prepare_input_dict_or_list(config)

    kwargs = _prepare_input_dict_or_list(kwargs)

    # Structured Config always converted first to OmegaConf
    if is_structured_config(config) or isinstance(config, (dict, list)):
        config = OmegaConf.structured(config, flags={"allow_objects": True})

    if OmegaConf.is_dict(config):
        # Finalize config (convert targets to strings, merge with kwargs)
        config_copy = copy.deepcopy(config)
        config_copy._set_flag(flags=["allow_objects", "struct", "readonly"], values=[True, False, False])
        config_copy._set_parent(config._get_parent())
        config = config_copy

        if kwargs:
            config = OmegaConf.merge(config, kwargs)

        OmegaConf.resolve(config)

        _partial_ = config.pop(_Keys.PARTIAL, False)

        return instantiate_node(config, *args, partial=_partial_, mode=mode)
    elif OmegaConf.is_list(config):
        # Finalize config (convert targets to strings, merge with kwargs)
        config_copy = copy.deepcopy(config)
        config_copy._set_flag(flags=["allow_objects", "struct", "readonly"], values=[True, False, False])
        config_copy._set_parent(config._get_parent())
        config = config_copy

        OmegaConf.resolve(config)

        _partial_ = kwargs.pop(_Keys.PARTIAL, False)

        if _partial_:
            raise InstantiationException("The _partial_ keyword is not compatible with top-level list instantiation")

        return instantiate_node(config, *args, partial=_partial_, mode=mode)
    else:
        raise InstantiationException(
            dedent(
                f"""\
                Cannot instantiate config of type {type(config).__name__}.
                Top level config must be an OmegaConf DictConfig/ListConfig object,
                a plain dict/list, or a Structured Config class or instance."""
            )
        )


def _locate(path: str) -> Any:
    """
    Locate an object by name or dotted path, importing as necessary.
    This function attempts to import modules starting from the most specific path
    (back to front), making it possible to import objects where the final component
    could be either a module or an attribute of the previous module.
    """
    if path == "":
        raise ImportError("Empty path")
    from importlib import import_module

    parts = [part for part in path.split(".")]
    for part in parts:
        if not len(part):
            raise ValueError(f"Error loading '{path}': invalid dotstring." + "\nRelative imports are not supported.")
    assert len(parts) > 0

    # Try importing from the most specific path first (back to front)
    for i in range(len(parts), 0, -1):
        module_path = ".".join(parts[:i])
        try:
            obj = import_module(module_path)

            # If this isn't the full path, get the remaining attributes
            remaining_parts = parts[i:]
            for part in remaining_parts:
                try:
                    obj = getattr(obj, part)
                except AttributeError as exc_attr:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_attr)}"
                        + f"\nAre you sure that '{part}' is an attribute of '{module_path}'?"
                    ) from exc_attr

            # Successfully found the object
            return obj

        except ModuleNotFoundError:
            # Module not found, try a less specific path
            continue
        except Exception as exc_import:
            # If we hit a different exception, it's likely an issue with the module itself
            raise ImportError(f"Error loading '{path}':\n{repr(exc_import)}") from exc_import

    # If we've tried all paths and nothing worked, report failure with the base module
    raise ImportError(
        f"Error loading '{path}': Unable to import any module in the path. "
        f"Are you sure that module '{parts[0]}' is installed?"
    )


def _is_target(x: Any) -> bool:
    if isinstance(x, dict):
        return "_target_" in x
    if OmegaConf.is_dict(x):
        return "_target_" in x
    return False


def _call_target(
    _target_: Callable[..., Any],
    _partial_: bool,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    full_key: str,
) -> Any:
    """Call target (type) with args and kwargs."""
    args, kwargs = _extract_pos_args(args, kwargs)

    # dont pass dataclasses fields init=False set
    if dataclasses.is_dataclass(_target_):
        not_init_field_names = {
            f.name for f in dataclasses.fields(_target_) if not f.init
        }
        # Filter the incoming kwargs against the valid init field names
        kwargs = {key: value for key, value in kwargs.items()
            if key not in not_init_field_names}

    if _partial_:
        try:
            return functools.partial(_target_, *args, **kwargs)
        except Exception as e:
            msg = f"Error in creating partial({_convert_target_to_string(_target_)}, ...) object:" + f"\n{repr(e)}"
            if full_key:
                msg += f"\nfull_key: {full_key}"
            raise InstantiationException(msg) from e
    else:
        try:
            return _target_(*args, **kwargs)
        except Exception as e:
            msg = f"Error in call to target '{_convert_target_to_string(_target_)}':\n{repr(e)}"
            if full_key:
                msg += f"\nfull_key: {full_key}"
            raise InstantiationException(msg) from e


def _convert_target_to_string(t: Any) -> Any:
    if callable(t):
        return f"{t.__module__}.{t.__qualname__}"
    else:
        return t


def _prepare_input_dict_or_list(d: Union[Dict[Any, Any], List[Any]]) -> Any:
    res: Any
    if isinstance(d, dict):
        res = {}
        for k, v in d.items():
            if k == "_target_":
                v = _convert_target_to_string(d["_target_"])
            elif isinstance(v, (dict, list)):
                v = _prepare_input_dict_or_list(v)
            res[k] = v
    elif isinstance(d, list):
        res = []
        for v in d:
            if isinstance(v, (list, dict)):
                v = _prepare_input_dict_or_list(v)
            res.append(v)
    else:
        assert False
    return res


def _resolve_target(
    target: Union[str, type, Callable[..., Any]],
    full_key: str,
    check_callable: bool = True,
) -> Union[type, Callable[..., Any], object]:
    """Resolve target string, type or callable into type or callable."""
    if isinstance(target, str):
        try:
            target = _locate(target)
        except Exception as e:
            msg = f"Error locating target '{target}'."
            if full_key:
                msg += f"\nfull_key: {full_key}"
            raise InstantiationException(msg) from e
    if check_callable and not callable(target):
        msg = f"Expected a callable target, got '{target}' of type '{type(target).__name__}'"
        if full_key:
            msg += f"\nfull_key: {full_key}"
        raise InstantiationException(msg)
    return target


def _extract_pos_args(input_args: Any, kwargs: Any) -> Tuple[Any, Any]:
    config_args = kwargs.pop(_Keys.ARGS, ())
    output_args = config_args

    if isinstance(config_args, Sequence):
        if len(input_args) > 0:
            output_args = input_args
    else:
        raise InstantiationException(
            f"Unsupported _args_ type: '{type(config_args).__name__}'. value: '{config_args}'"
        )

    return output_args, kwargs


def _convert_node(node: Any) -> Any:
    if OmegaConf.is_config(node):
        node = OmegaConf.to_container(node, resolve=True)

    return node


def instantiate_node(
    node: Any,
    *args: Any,
    partial: bool = False,
    mode: InstantiationMode = InstantiationMode.LENIENT,
) -> Any:
    # Return None if config is None
    if node is None or (OmegaConf.is_config(node) and node._is_none()):
        return None

    if not OmegaConf.is_config(node):
        return node

    # Override parent modes from config if specified
    if OmegaConf.is_dict(node):
        # using getitem instead of get(key, default) because OmegaConf will raise an exception
        # if the key type is incompatible on get.
        partial = node[_Keys.PARTIAL] if _Keys.PARTIAL in node else partial

    full_key = node._get_full_key(None)

    if not isinstance(partial, bool):
        msg = f"Instantiation: _partial_ flag must be a bool, got {type(partial)}"
        if node and full_key:
            msg += f"\nfull_key: {full_key}"
        raise TypeError(msg)

    if OmegaConf.is_list(node):
        items = [instantiate_node(item, mode=mode) for item in node._iter_ex(resolve=True)]

        return items
    elif OmegaConf.is_dict(node):
        exclude_keys = set(item.value for item in _Keys if item != _Keys.ARGS)
        if _is_target(node):
            should_call_target = node.get("_call_", True)
            _target_ = _resolve_target(node.get(_Keys.TARGET), full_key, check_callable=should_call_target)
            kwargs = {}
            is_partial = node.get("_partial_", False) or partial

            if not should_call_target:
                if len(set(node.keys()) - {"_target_", "_call_"}) != 0:
                    extra_keys = set(node.keys()) - {"_target_", "_call_"}
                    raise InstantiationException(
                        f"_call_ was set to False for target {_convert_target_to_string(_target_)}, but extra keys were found: {extra_keys}"
                    )
                else:
                    return _target_

            for key in node.keys():
                if key not in exclude_keys:
                    if OmegaConf.is_missing(node, key) and is_partial:
                        continue
                    value = node[key]
                    try:
                        value = instantiate_node(value, mode=mode)
                    except (ImportError, InstantiationException) as e:
                        if mode == InstantiationMode.STRICT:
                            raise InstantiationException(f"Error instantiating {value} for key {full_key}: {e}") from e
                        else:
                            value = None
                            logging.warning(
                                f"Error instantiating {value} for key {full_key}.{key}. Using None instead in lenient mode."
                            )
                    kwargs[key] = _convert_node(value)

            assert callable(_target_)
            return _call_target(_target_, partial, args, kwargs, full_key)
        else:
            dict_items = {}
            for key, value in node.items():
                dict_items[key] = instantiate_node(value, mode=mode)
            return dict_items

    else:
        assert False, f"Unexpected config type : {type(node).__name__}"
