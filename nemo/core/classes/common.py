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
from abc import ABC, abstractmethod
from typing import Dict, Optional

import hydra
import wrapt
from omegaconf import DictConfig, OmegaConf

from nemo.core.neural_types import NeuralType, NeuralTypeComparisonResult

__all__ = ['Typing', 'FileIO', 'Model', 'Serialization', 'typecheck']


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

    def _validate_input_types(self, **kwargs):
        # TODO: Properly implement this
        if self.input_types is not None:
            if len(kwargs) != len(self.input_types):
                raise TypeError(
                    "Number of input arguments provided ({}) is not as expected ({})".format(
                        len(kwargs), len(self.input_types)
                    )
                )

            for key, value in kwargs.items():
                # Check if keys exists in the defined input types
                if key not in self.input_types:
                    raise TypeError(
                        f"Input argument {key} has no corresponding input_type match. "
                        f"Existing input_types = {self.input_types.keys()}"
                    )

                # Perform neural type check
                if (
                    hasattr(value, 'neural_type')
                    and self.input_types[key].compare(value.neural_type) != NeuralTypeComparisonResult.SAME
                ):
                    raise TypeError(
                        f"{self.input_types[key].compare(value.neural_type)} : \n"
                        f"Input type expected = {self.input_types[key]} | \n"
                        f"Input type found : {value.neural_type}"
                    )

    def _attach_and_validate_output_types(self, out_objects):
        # TODO: Properly implement this
        if self.output_types is not None:
            out_types_list = list(self.output_types.items())
            if not isinstance(out_objects, tuple) and not isinstance(out_objects, list):
                out_objects.neural_type = out_types_list[0][1]
            else:
                for ind, res in enumerate(out_objects):
                    res.neural_type = out_types_list[ind][1]


class Serialization(ABC):
    @classmethod
    def from_config_dict(cls, config: DictConfig):
        """Instantiates object using DictConfig-based configuration"""
        if 'cls' in config:
            instance = hydra.utils.instantiate(config=config)
        else:
            instance = cls(cfg=config)
        if not hasattr(instance, '_cfg'):
            instance._cfg = config
        return instance

    def to_config_dict(self) -> DictConfig:
        """Returns object's configuration to config dictionary"""
        if hasattr(self, '_cfg') and self._cfg is not None:
            return self._cfg
        else:
            raise NotImplementedError(
                'to_config_dict() can currently only return object._cfg but current object does not have it.'
            )


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
        if issubclass(cls, Serialization):
            conf = OmegaConf.load(path2yaml_file)
            return cls.from_config_dict(config=conf)
        else:
            raise NotImplementedError()

    def to_config_file(self, path2yaml_file: str):
        """
        Saves current instance's configuration to YAML config file. Weights will not be saved.
        Args:
            path2yaml_file: path2yaml_file: path to yaml file where model model configuration will be saved

        Returns:
        """
        if hasattr(self, '_cfg'):
            with open(path2yaml_file, 'w') as fout:
                OmegaConf.save(config=self._cfg, f=fout, resolve=True)
        else:
            raise NotImplementedError()


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
        """
        A decorator which performs input-output neural type checks, and attaches
        neural types to the output of the function that it wraps.

        Requires that the class inherit from `nemo.core.Typing` in order to perform
        type checking, and will raise an error if that is not the case.

        # Usage
        @typecheck()
        def fn(self, arg1, arg2, ...):
            ...

        Points to be noted:
        1) The brackets () in `@typecheck()` are necessary.

            You will encounter a TypeError: __init__() takes 1 positional argument but X
            were given without those brackets.

        2) The function can take any number of positional arguments during definition.

            When you call this function, all arguments must be passed using kwargs only.

        """

    @wrapt.decorator
    def __call__(self, wrapped, instance: Typing, args, kwargs):
        if instance is None:
            raise RuntimeError("Only classes which inherit nemo.core.Typing can use this decorator !")

        if not isinstance(instance, Typing):
            raise RuntimeError("Only classes which inherit nemo.core.Typing can use this decorator !")

        # If types are not defined, skip type checks and just call the wrapped method
        if instance.input_types is None and instance.output_types is None:
            return wrapped(*args, **kwargs)

        # Check that all arguments are kwargs
        if instance.input_types is not None and len(args) > 0:
            raise TypeError("All arguments must be passed by kwargs only for typed methods")

        # Perform rudimentary input checks here
        instance._validate_input_types(**kwargs)

        # Call the method - this can be forward, or any other callable method
        outputs = wrapped(*args, **kwargs)

        instance._attach_and_validate_output_types(outputs)

        return outputs
