# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
__all__ = ['Typing', 'FileIO', 'Model', 'Serialization']

import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import wrapt
from ruamel.yaml import YAML

from nemo.core.neural_types import NeuralType, NeuralTypeComparisonResult
from nemo.utils import logging


class Typing(ABC):
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


class Serialization(ABC):
    @staticmethod
    def __instantiate_class_from_config(
        configuration: Dict[str, Any], name: str = None, overwrite_params: Dict[str, Any] = {}
    ):
        """
        Method instantiating the object based on the configuration (dictionary).
        Args:
            configuration: Dictionary containing proper "header" and "init_params" sections.
            name: name of the module that will overwrite the name in the `init_params` (optional, DEFAULT: None)
            overwrite_params: Dictionary containing parameters that will be added to or overwrite (!)
            the default init parameters loaded from the configuration file (the module "init_params" section).
        Returns:
            Instance of the created object.
        """

        def __class_from_header(serialized_header: Dict[str, Any]):
            """
            Args:
                Serialized_header: Dictionary containing module header.
            Returns:
                Class of the module to be created.
            """
            # Parse the "full specification".
            spec_list = serialized_header["full_spec"].split(".")

            # Get module class from the "full specification".
            mod_obj = __import__(spec_list[0])
            for spec in spec_list[1:]:
                mod_obj = getattr(mod_obj, spec)

            return mod_obj

        # Deserialize header - get object class.
        module_class = __class_from_header(configuration["header"])

        # Update parameters with additional ones.
        configuration["init_params"].update(overwrite_params)

        # Override module name in init_params using the logic:
        #  * section_name if not none overrides init_params.name first (skipped for now, TOTHINK!)
        #  * name (if None) overrides init_params.name
        if name is not None:
            configuration["init_params"]["name"] = name

        # Get init parameters.
        init_params = configuration["init_params"]

        # Create the module instance.
        new_module = module_class(**init_params)
        logging.info(f"Instantiated a new Neural Module of type {type(new_module).__name__}")

        # Return the module instance.
        return new_module

    @classmethod
    def from_config_dict(cls, configuration: Dict[str, Any]):
        """Instantiates object using dictionary-based configuration"""
        return cls.__instantiate_class_from_config(configuration=configuration)

    def to_config_dict(self) -> Dict[str, Any]:
        """Saves object's configuration to config dictionary"""
        # TODO: Implement me here
        pass


class FileIO(ABC):
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
        # TODO: implement me
        pass


class Model(Typing, Serialization, FileIO):
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
    def from_pretrained(cls, name: str):
        """
        Instantiates an instance of NeMo from NVIDIA NGC cloud
        Args:
            name: string key which will be used to find the module. Could be path to local .nemo file.

        Returns:
            A model instance of a class derived from INMModelAPI
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


class typecheck:

    def __init__(self):
        pass

    @wrapt.decorator
    def __call__(self, wrapped, instance: Typing, args, kwargs):
        if instance is None:
            raise RuntimeError("Only classes which inherit nemo.core.Typing can use this decorator !")

        if not (hasattr(instance, 'input_types') and hasattr(instance, 'output_types')):
            raise RuntimeError("Only classes which inherit nemo.core.Typing can use this decorator !")

        # If types are not defined, skip type checks and just call the
        if instance.input_types is None and instance.output_types is None:
            return wrapped(*args, **kwargs)

        # Check that all arguments are kwargs
        if len(args) > 0:
            raise TypeError("All arguments must be passed by kwargs only for typed methods")

        # Perform rudimentary input checks here
        self._validate_input_types(instance, kwargs)

        # Call the method - this can be forward, or any other callable method
        outputs = wrapped(*args, **kwargs)

        self._attach_and_validate_output_types(instance, outputs)

        return outputs

    def _validate_input_types(self, instance, kwargs):
        # TODO: Properly implement this
        if instance.input_types is not None:
            if len(kwargs) != len(instance.input_types):
                raise TypeError("Number of input arguments provided ({}) is not as expected ({})".format(
                    len(kwargs), len(instance.input_types))
                )

            for key, value in kwargs.items():
                if (
                    hasattr(value, 'neural_type')
                    and instance.input_types[key].compare(value.neural_type) != NeuralTypeComparisonResult.SAME
                ):
                    raise TypeError(f"{instance.input_types[key].compare(value.neural_type)}")

    def _attach_and_validate_output_types(self, instance, out_objects):
        # TODO: Properly implement this
        if instance.output_types is not None:
            out_types_list = list(instance.items())
            if not isinstance(out_objects, tuple) and not isinstance(out_objects, list):
                out_objects.neural_type = out_types_list[0][1]
            else:
                for ind, res in enumerate(out_objects):
                    res.neural_type = out_types_list[ind][1]
