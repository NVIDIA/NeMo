# Copyright (c) 2019-, NVIDIA CORPORATION. All rights reserved.
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
"""Interfaces common to all Neural Modules and Neural Modules."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ruamel.yaml import YAML

from nemo.core.neural_types import NeuralType
from nemo.utils import instantiate_class_from_config

__all__ = ['NeuralModuleAPI', 'NeuralModelAPI']


class NeuralModuleAPI(ABC):
    """
    Abstract class offering interface shared between all Neural Modules.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """Define these to enable input neural type checks"""
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Define these to enable output neural type checks"""
        return None

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
        return instantiate_class_from_config(configuration)

    def to_config_dict(self) -> Dict[str, Any]:
        """Saves object's configuration to config dictionary"""
        # TODO: Implement me here
        raise NotImplementedError()


class NeuralModelAPI(NeuralModuleAPI):
    """
    Abstract class offering interface shared between all Neural Models.
    """

    @classmethod
    @abstractmethod
    def from_cloud(cls, name: str):
        """
        Instantiates an instance of Neural Model from NVIDIA NGC cloud
        Args:
            name: string key which will be used to find the module

        Returns:
            A model instance of a class derived from NeuralModelAPI
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

    @abstractmethod
    def setup_training_data(self, train_data_layer_params: Optional[Dict]):
        """
        Setups data loader to be used in training
        Args:
            train_data_layer_params: training data layer parameters.
        Returns:

        """
        pass

    @abstractmethod
    def setup_training_data(self, val_data_layer_params: Optional[Dict]):
        """
        (Optionally) Setups data loader to be used in validation
        Args:
            val_data_layer_params: validation data layer parameters.
        Returns:

        """
        pass

    @abstractmethod
    def setup_test_data(self, test_data_layer_params: Optional[Dict]):
        """
        (Optionally) Setups data loader to be used in testing
        Args:
            test_data_layer_params: test data layer parameters.
        Returns:

        """
        pass

    @abstractmethod
    def setup_optimization(self, optim_params: Optional[Dict]):
        """
        Setups optimization parameters
        Args:
            optim_params: dictionary with optimization parameters.
        Returns:

        """
        pass

    @classmethod
    def from_config_file(cls, path2yaml_file: str):
        """
        Instantiates an instance of Neural Model from YAML config file.
        Weights will be initialized randomly.
        Args:
            path2yaml_file: path to yaml file with model configuration

        Returns:

        """
        yaml = YAML(typ="safe")
        with open(path2yaml_file) as f:
            model_config_dict = yaml.load(f)
            instance = NeuralModuleAPI.from_config_dict(configuration=model_config_dict)
            return instance

    def to_config_file(self, path2yaml_file: str):
        """
        Saves current instance's configuration to YAML config file. Weights will not be saved.
        Args:
            path2yaml_file: path2yaml_file: path to yaml file where model model configuration will be saved

        Returns:

        """
        raise NotImplementedError()
