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

import importlib
import os
import pkgutil
import sys
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from typing import Any, Dict, Optional

import attr
import attrs
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from cosmos1.utils import log
from cosmos1.utils.config import Config


def is_attrs_or_dataclass(obj) -> bool:
    """
    Check if the object is an instance of an attrs class or a dataclass.

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is an instance of an attrs class or a dataclass, False otherwise.
    """
    return is_dataclass(obj) or attr.has(type(obj))


def get_fields(obj):
    """
    Get the fields of an attrs class or a dataclass.

    Args:
        obj: The object to get fields from. Must be an instance of an attrs class or a dataclass.

    Returns:
        list: A list of field names.

    Raises:
        ValueError: If the object is neither an attrs class nor a dataclass.
    """
    if is_dataclass(obj):
        return [field.name for field in dataclass_fields(obj)]
    elif attr.has(type(obj)):
        return [field.name for field in attr.fields(type(obj))]
    else:
        raise ValueError("The object is neither an attrs class nor a dataclass.")


def override(config: Config, overrides: Optional[list[str]] = None) -> Config:
    """
    :param config: the instance of class `Config` (usually from `make_config`)
    :param overrides: list of overrides for config
    :return: the composed instance of class `Config`
    """
    # Store the class of the config for reconstruction after overriding.
    # config_class = type(config)

    # Convert Config object to a DictConfig object
    config_dict = attrs.asdict(config)
    config_omegaconf = DictConfig(content=config_dict, flags={"allow_objects": True})
    # Enforce "--" separator between the script arguments and overriding configs.
    if overrides:
        if overrides[0] != "--":
            raise ValueError('Hydra config overrides must be separated with a "--" token.')
        overrides = overrides[1:]
    # Use Hydra to handle overrides
    cs = ConfigStore.instance()
    cs.store(name="config", node=config_omegaconf)
    with initialize(version_base=None):
        config_omegaconf = compose(config_name="config", overrides=overrides)
        OmegaConf.resolve(config_omegaconf)

    def config_from_dict(ref_instance: Any, kwargs: Any) -> Any:
        """
        Construct an instance of the same type as ref_instance using the provided dictionary or data or unstructured data

        Args:
            ref_instance: The reference instance to determine the type and fields when needed
            kwargs: A dictionary of keyword arguments to use for constructing the new instance or primitive data or unstructured data

        Returns:
            Any: A new instance of the same type as ref_instance constructed using the provided kwargs or the primitive data or unstructured data

        Raises:
            AssertionError: If the fields do not match or if extra keys are found.
            Exception: If there is an error constructing the new instance.
        """
        is_type = is_attrs_or_dataclass(ref_instance)
        if not is_type:
            return kwargs
        else:
            ref_fields = set(get_fields(ref_instance))
            assert isinstance(kwargs, dict) or isinstance(
                kwargs, DictConfig
            ), "kwargs must be a dictionary or a DictConfig"
            keys = set(kwargs.keys())

            # ref_fields must equal to or include all keys
            extra_keys = keys - ref_fields
            assert ref_fields == keys or keys.issubset(
                ref_fields
            ), f"Fields mismatch: {ref_fields} != {keys}. Extra keys found: {extra_keys} \n \t when constructing {type(ref_instance)} with {keys}"

            resolved_kwargs: Dict[str, Any] = {}
            for f in keys:
                resolved_kwargs[f] = config_from_dict(getattr(ref_instance, f), kwargs[f])
            try:
                new_instance = type(ref_instance)(**resolved_kwargs)
            except Exception as e:
                log.error(f"Error when constructing {type(ref_instance)} with {resolved_kwargs}")
                log.error(e)
                raise e
            return new_instance

    config = config_from_dict(config, config_omegaconf)

    return config


def get_config_module(config_file: str) -> str:
    if not config_file.endswith(".py"):
        log.error("Config file cannot be specified as module.")
        log.error("Please provide the path to the Python config file (relative to the Cosmos root).")
    assert os.path.isfile(config_file), f"Cosmos config file ({config_file}) not found."
    # Convert to importable module format.
    config_module = config_file.replace("/", ".").replace(".py", "")
    return config_module


def import_all_modules_from_package(package_path: str, reload: bool = False, skip_underscore: bool = True) -> None:
    """
    Import all modules from the specified package path recursively.

    This function is typically used in conjunction with Hydra to ensure that all modules
    within a specified package are imported, which is necessary for registering configurations.

    Example usage:
    ```python
    import_all_modules_from_package("cosmos1.models.diffusion.config.inference", reload=True, skip_underscore=False)
    ```

    Args:
        package_path (str): The dotted path to the package from which to import all modules.
        reload (bool): Flag to determine whether to reload modules if they're already imported.
        skip_underscore (bool): If True, skips importing modules that start with an underscore.
    """
    log.debug(f"{'Reloading' if reload else 'Importing'} all modules from package {package_path}")
    package = importlib.import_module(package_path)
    package_directory = package.__path__

    def import_modules_recursively(directory: str, prefix: str) -> None:
        """
        Recursively imports or reloads all modules in the given directory.

        Args:
            directory (str): The file system path to the current package directory.
            prefix (str): The module prefix (e.g., 'cosmos1.models.diffusion.config').
        """
        for _, module_name, is_pkg in pkgutil.iter_modules([directory]):
            if skip_underscore and module_name.startswith("_"):
                log.debug(f"Skipping module {module_name} as it starts with an underscore")
                continue

            full_module_name = f"{prefix}.{module_name}"
            log.debug(f"{'Reloading' if reload else 'Importing'} module {full_module_name}")

            if full_module_name in sys.modules and reload:
                importlib.reload(sys.modules[full_module_name])
            else:
                importlib.import_module(full_module_name)

            if is_pkg:
                sub_package_directory = os.path.join(directory, module_name)
                import_modules_recursively(sub_package_directory, full_module_name)

    for directory in package_directory:
        import_modules_recursively(directory, package_path)
