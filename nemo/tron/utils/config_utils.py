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
import os
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from typing import Any, Optional, Type, TypeVar

import yaml
from omegaconf import OmegaConf

from nemo.tron.utils.instantiate_utils import InstantiationMode, instantiate
from nemo.tron.utils.yaml_utils import safe_yaml_representers

T = TypeVar("T", bound="ConfigContainer")


@dataclass(kw_only=True)
class ConfigContainer:
    """
    Base configuration container for NeMo Tron configurations.

    Provides:
    - Custom validation
    - Versioning metadata
    - YAML/Dict serialization and deserialization
    - Dictionary-style attribute access (config["attr"] and config.get("attr", default))
    """

    __version__: str = "0.1.0"

    @classmethod
    def from_dict(
        cls: Type[T],
        config_dict: dict[str, Any],
        mode: InstantiationMode = InstantiationMode.STRICT,
    ) -> T:
        """
        Create a config container from a dictionary using instantiate.

        Args:
            config_dict: Dictionary containing configuration
            mode: Serialization mode (strict or lenient)

        Returns:
            A new instance of this class initialized with the dictionary values
        """
        # Make a copy to avoid modifying the input
        config_dict = copy.deepcopy(config_dict)

        assert "_target_" in config_dict

        # Check for extra keys in strict mode
        expected_fields = {f.name for f in dataclass_fields(cls) if not f.name.startswith("_")}
        expected_fields.add("_target_")  # Add _target_ as a valid field
        extra_keys = set(config_dict.keys()) - expected_fields

        if extra_keys:
            if mode == InstantiationMode.STRICT:
                raise ValueError(f"Dictionary contains extra keys not in {cls.__qualname__}: {extra_keys}")
            else:
                # In lenient mode, remove extra keys
                for key in extra_keys:
                    config_dict.pop(key)

        # Use instantiate to create the object
        instance = instantiate(config_dict, mode=mode)

        return instance

    @classmethod
    def from_yaml(cls: Type[T], yaml_path: str, mode: InstantiationMode = InstantiationMode.LENIENT) -> T:
        """
        Create a config container from a YAML file.

        Args:
            yaml_path: Path to the YAML file
            mode: Serialization mode (strict or lenient)

        Returns:
            A new instance of this class initialized with the YAML file values
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Convert to OmegaConf first for better compatibility with instantiate
        conf = OmegaConf.create(config_dict)

        return cls.from_dict(OmegaConf.to_container(conf, resolve=True), mode=mode)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the config container to a dictionary.

        Also converts any nested dataclasses (both ConfigContainer and regular dataclasses)
        to dictionaries recursively.

        Returns:
            Dictionary representation of this config
        """
        result = {}
        result["_target_"] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"

        for f in dataclass_fields(self):
            if f.name.startswith("_"):
                continue

            value = getattr(self, f.name)
            result[f.name] = self._convert_value_to_dict(value)

        return result

    @classmethod
    def _convert_value_to_dict(cls, value: Any) -> Any:
        """
        Recursively convert a value to a dictionary representation.

        Handles:
        - ConfigContainer instances (using to_dict)
        - Regular dataclasses (using asdict)
        - Lists and tuples (converting each element)
        - Dictionaries (converting each value)
        - Other types (kept as-is)

        Args:
            value: The value to convert

        Returns:
            The converted value
        """
        if isinstance(value, ConfigContainer):
            return value.to_dict()
        elif is_dataclass(value) and not isinstance(value, type):
            # Handle regular dataclasses
            result = {}

            # Add _target_ field for instantiation
            result["_target_"] = f"{value.__class__.__module__}.{value.__class__.__qualname__}"

            # Convert each field, handling nested dataclasses properly
            for field in dataclass_fields(value):
                if field.name.startswith("_"):
                    continue

                field_value = getattr(value, field.name)
                result[field.name] = cls._convert_value_to_dict(field_value)

            return result
        elif isinstance(value, (list, tuple)):
            return [cls._convert_value_to_dict(item) for item in value]
        elif isinstance(value, dict):
            return {k: cls._convert_value_to_dict(v) for k, v in value.items()}
        else:
            return value

    def to_yaml(self, yaml_path: Optional[str] = None) -> None:
        """
        Save the config container to a YAML file.

        Args:
            yaml_path: Path where to save the YAML file
        """
        config_dict = self.to_dict()

        with safe_yaml_representers():
            if yaml_path is None:
                print(yaml.safe_dump(config_dict, default_flow_style=False))
            else:
                with open(yaml_path, "w") as f:
                    yaml.safe_dump(config_dict, f, default_flow_style=False)

    def __deepcopy__(self, memo):
        """Support for deep copying."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for f in dataclass_fields(self):
            setattr(result, f.name, copy.deepcopy(getattr(self, f.name), memo))

        return result
