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
from typing import Any, Callable, Sequence, Union

from omegaconf import OmegaConf
from omegaconf._utils import is_structured_config


class InstantiationException(Exception):
    """Custom exception type for instantiation errors."""

    ...


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
    """Instantiate an object or callable from a config object.

    This function takes a configuration object (dictionary, list, OmegaConf config,
    or Structured Config instance) and instantiates the target specified within it.

    The config object must contain:
        _target_ (str): The fully qualified name of the class or callable to instantiate.

    The config object may also contain:
        _args_ (list): Positional arguments for the target.
        _partial_ (bool): If True, return a functools.partial object instead of calling
                         the target. Defaults to False.
        _call_ (bool): If False, simply resolves and returns the target without calling it.
                       Defaults to True.
        Additional keyword arguments to pass to the target.

    Args:
        config: The configuration object describing the target and its parameters.
        *args: Optional positional arguments that will override _args_ in the config
               if provided.
        mode: Instantiation mode (STRICT or LENIENT). In LENIENT mode (default),
              errors during instantiation of parameters are logged as warnings,
              and None is used instead. In STRICT mode, errors are raised.
        **kwargs: Optional keyword arguments that will override parameters in the config.
                  Note: Dataclass instances in kwargs are treated as nested configs.

    Returns:
        The instantiated object or the return value of the callable.
        If config._partial_ is True, returns a functools.partial object.
        If config._call_ is False, returns the resolved target callable/class itself.
        Returns None if the input config is None.

    Raises:
        InstantiationException: If the config is invalid, the target cannot be resolved,
                                or instantiation fails in STRICT mode.
        TypeError: If the _partial_ flag is not a boolean.
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


def instantiate_node(
    node: Any,
    *args: Any,
    partial: bool = False,
    mode: InstantiationMode = InstantiationMode.LENIENT,
) -> Any:
    """Recursively instantiates a node within a configuration structure.

    This function handles the instantiation of individual nodes (dictionaries,
    lists, or primitive values) within a larger configuration tree, typically
    managed by OmegaConf.

    If the node is a dictionary containing a `_target_` key, it resolves and
    instantiates the target callable/class using the other items in the
    dictionary as keyword arguments. Nested nodes are recursively instantiated.

    If the node is a list, it recursively instantiates each item in the list.

    If the node is not an OmegaConf config node (e.g., a primitive type), it's
    returned directly.

    Args:
        node: The configuration node to instantiate (can be DictConfig, ListConfig,
              or a primitive type).
        *args: Positional arguments passed down from the top-level `instantiate` call,
               used primarily for the final target call if the node is a dictionary
               with `_target_`.
        partial: Boolean flag indicating whether to return a `functools.partial` object
                 instead of calling the target. This can be overridden by a
                 `_partial_` key within the node itself.
        mode: Instantiation mode (STRICT or LENIENT). Determines error handling
              behavior for nested instantiations.

    Returns:
        The instantiated object, list, or the original node if it wasn't a config.
        Returns None if the input node is None or represents a None value in OmegaConf.

    Raises:
        InstantiationException: If instantiation fails in STRICT mode, or if there are
                                issues like incompatible arguments or non-callable targets.
        TypeError: If a `_partial_` flag within the config is not a boolean.
    """
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
                        f"_call_ was set to False for target {_convert_target_to_string(_target_)},"
                        f" but extra keys were found: {extra_keys}"
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
                                f"Error instantiating {value} for key {full_key}.{key}. "
                                f"Using None instead in lenient mode."
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
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    full_key: str,
) -> Any:
    """Call target (type) with args and kwargs."""
    args, kwargs = _extract_pos_args(args, kwargs)
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


def _prepare_input_dict_or_list(d: Union[dict[Any, Any], list[Any]]) -> Any:
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


def _extract_pos_args(input_args: Any, kwargs: Any) -> tuple[Any, Any]:
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
