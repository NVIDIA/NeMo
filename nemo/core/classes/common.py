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


_TYPECHECK_ENABLED = True


def is_typecheck_enabled():
    """
    Getter method for typechecking state.
    """
    return _TYPECHECK_ENABLED


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
        """
        This function does a few things.
        1) It ensures that len(kwargs) == len(self.input_types).
        2) If above fails, it checks len(kwargs) == len(self.input_types <non-optional>).
        3) For each (keyword name, keyword value) passed as input to the wrapped function:
            - Check if the keyword name exists in the list of valid self.input_types names.
            - Check if keyword value has the `neural_type` property.
                - If it does, then perform a comparative check and assert that neural types
                    are compatible (SAME or GREATER).
            - Check if keyword value is a container type (list or tuple). If yes,
                then perform the elementwise test of neural type above on each element
                of the nested structure, recursively.

        Args:
            kwargs: Dictionary of argument_name:argument_value pairs passed to the wrapped
                function upon call.
        """
        # TODO: Properly implement this
        if self.input_types is not None:
            total_input_types = len(self.input_types)
            mandatory_input_types = len(
                [type_val for type_key, type_val in self.input_types.items() if not type_val.optional]
            )

            if len(kwargs) != total_input_types:
                if len(kwargs) != mandatory_input_types:
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
                if hasattr(value, 'neural_type') and not self.input_types[key].compare(value.neural_type) in (
                    NeuralTypeComparisonResult.SAME,
                    NeuralTypeComparisonResult.GREATER,
                ):
                    raise TypeError(
                        f"{self.input_types[key].compare(value.neural_type)} : \n"
                        f"Input type expected = {self.input_types[key]} | \n"
                        f"Input type found : {value.neural_type}"
                    )

                # Perform recursive neural type check for homogeneous elements
                elif isinstance(value, list) or isinstance(value, tuple):
                    for ind, val in enumerate(value):
                        self.__check_neural_type(val, self.input_types[key])

    def _attach_and_validate_output_types(self, out_objects):
        """
        This function does a few things.
        1) It ensures that len(out_object) == len(self.output_types).
        2) If the output is a tensor (or list/tuple of list/tuple ... of tensors), it
            attaches a neural_type to it. For objects without the neural_type attribute,
            such as python objects (dictionaries and lists, primitive data types, structs),
            no neural_type is attached.

            Note: tensor.neural_type is only checked during _validate_input_types which is
            called prior to forward().

        Args:
            out_objects: The outputs of the wrapped function.
        """
        # TODO: Properly implement this
        if self.output_types is not None:
            out_types_list = list(self.output_types.items())

            # First convert all outputs to list/tuple format to check correct number of outputs
            if type(out_objects) in (list, tuple):
                out_container = out_objects
            else:
                out_container = [out_objects]

            if len(self.output_types) != len(out_container):
                raise TypeError(
                    "Number of output arguments provided ({}) is not as expected ({})".format(
                        len(out_container), len(self.output_types)
                    )
                )

            # Attach types recursively, if possible
            if not isinstance(out_objects, tuple) and not isinstance(out_objects, list):
                try:
                    out_objects.neural_type = out_types_list[0][1]
                except Exception:
                    pass
            else:
                for ind, res in enumerate(out_objects):
                    self.__attach_neural_type(res, out_types_list[ind][1])

    def __check_neural_type(self, obj, type_val):
        if isinstance(obj, tuple) or isinstance(obj, list):
            for elem in obj:
                self.__check_neural_type(elem, type_val)
            return  # after processing nest, return to avoid testing nest itself

        if hasattr(obj, 'neural_type') and not type_val.compare(obj.neural_type) in (
            NeuralTypeComparisonResult.SAME,
            NeuralTypeComparisonResult.GREATER,
        ):
            raise TypeError(
                f"{type_val.compare(obj.neural_type)} : \n"
                f"Input type expected = {type_val} | \n"
                f"Input type found : {obj.neural_type}"
            )

    def __attach_neural_type(self, obj, type_val):
        if isinstance(obj, tuple) or isinstance(obj, list):
            for elem in obj:
                self.__attach_neural_type(elem, type_val)
            return  # after processing nest, return to avoid argument insertion into nest itself

        try:
            obj.neural_type = type_val
        except Exception:
            pass


class Serialization(ABC):
    @classmethod
    def from_config_dict(cls, config: DictConfig):
        """Instantiates object using DictConfig-based configuration"""
        if 'cls' in config and 'params' in config:
            # regular hydra-based instantiation
            instance = hydra.utils.instantiate(config=config)
        else:
            # models are handled differently for now
            instance = cls(cfg=config)
        if not hasattr(instance, '_cfg'):
            instance._cfg = config
        return instance

    def to_config_dict(self) -> DictConfig:
        """Returns object's configuration to config dictionary"""
        if hasattr(self, '_cfg') and self._cfg is not None and isinstance(self._cfg, DictConfig):
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

    @wrapt.decorator(enabled=is_typecheck_enabled)
    def __call__(self, wrapped, instance: Typing, args, kwargs):
        if instance is None:
            raise RuntimeError("Only classes which inherit nemo.core.Typing can use this decorator !")

        if not isinstance(instance, Typing):
            raise RuntimeError("Only classes which inherit nemo.core.Typing can use this decorator !")

        if hasattr(instance, 'input_ports') or hasattr(instance, 'output_ports'):
            raise RuntimeError(
                "Typing requires override of `input_types()` and `output_types()`, "
                "not `input_ports() and `output_ports()`"
            )

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

    @staticmethod
    def set_typecheck_enabled(enabled: bool = True):
        global _TYPECHECK_ENABLED
        _TYPECHECK_ENABLED = enabled
