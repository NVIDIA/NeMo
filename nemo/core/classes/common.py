# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

"""Interfaces common to all Neural Modules and Models."""
__all__ = ['NeMoTyping', 'NeuralModuleIO', 'NeMoModelAPI']

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ruamel.yaml import YAML

from nemo.core.neural_types import NeuralType, NeuralTypeComparisonResult


class NeMoTyping(ABC):
    """
    An interface which endows module with neural types
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """Define these to enable input neural type checks"""
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Define these to enable output neural type checks"""
        return None

    def __validate_input_types(self, in_objects):
        # TODO: Properly implement this
        if self.input_types is not None:
            for key, value in in_objects.items():
                if (
                    hasattr(value, 'neural_type')
                    and self.input_types[key].compare(value.neural_type) != NeuralTypeComparisonResult.SAME
                ):
                    raise TypeError(f"{self.input_types[key].compare(value.neural_type)}")

    def __attach_and_validate_output_types(self, out_objects):
        # TODO: Properly implement this
        if self.output_types is not None:
            out_types_list = list(self.output_types.items())
            if not isinstance(out_objects, tuple) and not isinstance(out_objects, list):
                out_objects.neural_type = out_types_list[0][1]
            else:
                for ind, res in enumerate(out_objects):
                    res.neural_type = out_types_list[ind][1]


class NeuralModuleIO(ABC):
    @abstractmethod
    def save_to(self, save_path: str):
        """Saves module/model with weights"""
        pass

    @classmethod
    @abstractmethod
    def restore_from(cls, restore_path: str):
        """Restores module/model with weights"""
        pass

    @classmethod
    def from_config_dict(cls, configuration: Dict[str, Any]):
        """Instantiates object using dictionary-based configuration"""
        # TODO: Implement me here
        raise NotImplementedError()

    def to_config_dict(self) -> Dict[str, Any]:
        """Saves object's configuration to config dictionary"""
        # TODO: Implement me here
        raise NotImplementedError()


class NeMoModelAPI(NeMoTyping, NeuralModuleIO):
    """
    Abstract class offering interface which should be implemented by all NeMo models.
    """

    @classmethod
    @abstractmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        """
        Should list all pre-trained models available via NVIDIA NGC cloud

        Returns:
            A dictionary of NeMo model key name -> NGC wget URI
        """
        pass

    @classmethod
    @abstractmethod
    def from_cloud(cls, name: str):
        """
        Instantiates an instance of NeMo from NVIDIA NGC cloud
        Args:
            name: string key which will be used to find the module

        Returns:
            A model instance of a class derived from NeMoModelAPI
        """
        pass

    @abstractmethod
    def export(self, **kwargs):
        """
        Exports model for deployment
        Args:
            **kwargs:

        Returns:

        """

    @classmethod
    def from_config_file(cls, path2yaml_file: str):
        """
        Instantiates an instance of NeMo Model from YAML config file.
        Weights will be initialized randomly.
        Args:
            path2yaml_file: path to yaml file with model configuration

        Returns:

        """
        yaml = YAML(typ="safe")
        with open(path2yaml_file) as f:
            model_config_dict = yaml.load(f)
            instance = cls.from_config_dict(configuration=model_config_dict)
            return instance

    def to_config_file(self, path2yaml_file: str):
        """
        Saves current instance's configuration to YAML config file. Weights will not be saved.
        Args:
            path2yaml_file: path2yaml_file: path to yaml file where model model configuration will be saved

        Returns:

        """
        raise NotImplementedError()
