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

import enum
import functools
import inspect
from contextlib import contextmanager
from typing import Generator

import yaml


@contextmanager
def safe_yaml_representers() -> Generator[None, None, None]:
    """
    Context manager for safely adding and removing custom YAML representers.

    Temporarily adds custom representers for functions, classes, and other objects
    to the YAML SafeDumper, and restores the original representers when exiting
    the context.

    Usage:
        with safe_yaml_representers():
            yaml_str = yaml.safe_dump(my_complex_object)
    """
    # Save original representers
    original_representers = yaml.SafeDumper.yaml_representers.copy()
    original_multi_representers = yaml.SafeDumper.yaml_multi_representers.copy()

    try:
        # Register custom representers

        # Partial representer
        yaml.SafeDumper.add_representer(functools.partial, _partial_representer)

        # Enum representer
        yaml.SafeDumper.add_multi_representer(enum.Enum, _enum_representer)

        # Function representer
        yaml.SafeDumper.add_representer(type(lambda: ...), _function_representer)
        yaml.SafeDumper.add_representer(type(object), _function_representer)

        # Try to add torch dtype representer if available
        try:
            import torch

            yaml.SafeDumper.add_representer(torch.dtype, _torch_dtype_representer)
        except ModuleNotFoundError:
            pass

        # Try to add GenerationConfig representer if available
        try:
            from transformers import GenerationConfig

            yaml.SafeDumper.add_representer(GenerationConfig, _generation_config_representer)
        except ModuleNotFoundError:
            pass

        # General object representer
        yaml.SafeDumper.add_multi_representer(object, _safe_object_representer)

        yield
    finally:
        # Restore original representers
        yaml.SafeDumper.yaml_representers = original_representers
        yaml.SafeDumper.yaml_multi_representers = original_multi_representers


def _function_representer(dumper, data):
    """Represent functions in YAML."""
    value = {
        "_target_": f"{inspect.getmodule(data).__name__}.{data.__qualname__}",  # type: ignore
        "_call_": False,
    }
    return dumper.represent_data(value)


def _torch_dtype_representer(dumper, data):
    """Represent torch dtypes in YAML."""
    value = {
        "_target_": str(data),
        "_call_": False,
    }
    return dumper.represent_data(value)


def _safe_object_representer(dumper, data):
    """
    General object representer for YAML.

    This function is a fallback for objects that don't have specific representers.
    If the object has __qualname__ attr,
    the _target_ is set to f"{inspect.getmodule(obj).__name__}.{obj.__qualname__}".
    If the object does not have a __qualname__ attr, the _target_ is set from its __class__ attr.
    The _call_ key is used to indicate whether the target should be called to create an instance.

    Args:
        dumper (yaml.Dumper): The YAML dumper to use for serialization.
        data (Any): The data to serialize.

    Returns:
        The YAML representation of the data.
    """
    try:
        obj = data
        target = f"{inspect.getmodule(obj).__name__}.{obj.__qualname__}"
        call = False
    except AttributeError:
        obj = data.__class__
        target = f"{inspect.getmodule(obj).__name__}.{obj.__qualname__}"
        call = True

    value = {
        "_target_": target,  # type: ignore
        "_call_": call,
    }
    return dumper.represent_data(value)


def _partial_representer(dumper, data):
    """Represent functools.partial objects in YAML."""
    # Get the underlying function
    func = data.func

    # Create a dictionary representation
    value = {
        "_target_": f"{inspect.getmodule(func).__name__}.{func.__qualname__}",
        "_partial_": True,
        "_args_": list(data.args) if data.args else [],
    }

    # Add keyword arguments if any exist
    if data.keywords:
        for k, v in data.keywords.items():
            value[k] = v

    return dumper.represent_data(value)


def _enum_representer(dumper, data):
    """Represent enums in YAML."""
    # Create a dictionary representation
    enum_class = data.__class__
    value = {
        "_target_": f"{inspect.getmodule(enum_class).__name__}.{enum_class.__qualname__}",
        "_call_": True,
        "_args_": [data.value],
    }

    return dumper.represent_data(value)


def _generation_config_representer(dumper, data):
    """Represent transformers GenerationConfig objects in YAML."""
    cls = data.__class__
    value = {
        "_target_": f"{inspect.getmodule(cls).__name__}.{cls.__qualname__}.from_dict",
        "_call_": True,
        "config_dict": data.to_dict(),
    }

    return dumper.represent_data(value)
