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
import hashlib
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union

import hydra
import wrapt
from omegaconf import DictConfig, OmegaConf

import nemo
from nemo.core.neural_types import NeuralType, NeuralTypeComparisonResult
from nemo.utils import logging
from nemo.utils.cloud import maybe_download_from_cloud

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

    def _validate_input_types(self, input_types=None, **kwargs):
        """
        This function does a few things.
        1) It ensures that len(self.input_types <non-optional>) <= len(kwargs) <= len(self.input_types).
        2) For each (keyword name, keyword value) passed as input to the wrapped function:
            - Check if the keyword name exists in the list of valid self.input_types names.
            - Check if keyword value has the `neural_type` property.
                - If it does, then perform a comparative check and assert that neural types
                    are compatible (SAME or GREATER).
            - Check if keyword value is a container type (list or tuple). If yes,
                then perform the elementwise test of neural type above on each element
                of the nested structure, recursively.

        Args:
            input_types: Either the `input_types` defined at class level, or the local function
                overridden type definition.
            kwargs: Dictionary of argument_name:argument_value pairs passed to the wrapped
                function upon call.
        """
        # TODO: Properly implement this
        if input_types is not None:
            total_input_types = len(input_types)
            mandatory_input_types = len(
                [type_val for type_key, type_val in input_types.items() if not type_val.optional]
            )

            if len(kwargs) < mandatory_input_types or len(kwargs) > total_input_types:
                raise TypeError(
                    f"Number of input arguments provided ({len(kwargs)}) is not as expected. Function has "
                    f"{total_input_types} total inputs with {mandatory_input_types} mandatory inputs."
                )

            for key, value in kwargs.items():
                # Check if keys exists in the defined input types
                if key not in input_types:
                    raise TypeError(
                        f"Input argument {key} has no corresponding input_type match. "
                        f"Existing input_types = {input_types.keys()}"
                    )

                # Perform neural type check
                if hasattr(value, 'neural_type') and not input_types[key].compare(value.neural_type) in (
                    NeuralTypeComparisonResult.SAME,
                    NeuralTypeComparisonResult.GREATER,
                ):
                    raise TypeError(
                        f"{input_types[key].compare(value.neural_type)} : \n"
                        f"Input type expected = {input_types[key]} | \n"
                        f"Input type found : {value.neural_type}"
                    )

                # Perform input ndim check
                if hasattr(value, 'shape'):
                    value_shape = value.shape
                    type_shape = input_types[key].axes
                    name = key

                    if type_shape is not None and len(value_shape) != len(type_shape):
                        raise TypeError(
                            f"Input shape mismatch occured for {name} in module {self.__class__.__name__} : \n"
                            f"Input shape expected = {input_types[key].axes} | \n"
                            f"Input shape found : {value_shape}"
                        )

                # Perform recursive neural type check for homogeneous elements
                elif isinstance(value, list) or isinstance(value, tuple):
                    for ind, val in enumerate(value):
                        self.__check_neural_type(val, input_types[key], name=key)

    def _attach_and_validate_output_types(self, out_objects, output_types=None):
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
            output_types: Either the `output_types` defined at class level, or the local function
                overridden type definition.
            out_objects: The outputs of the wrapped function.
        """
        # TODO: Properly implement this
        if output_types is not None:
            out_types_list = list(output_types.items())

            # First convert all outputs to list/tuple format to check correct number of outputs
            if type(out_objects) in (list, tuple):
                out_container = out_objects
            else:
                out_container = [out_objects]

            if len(output_types) != len(out_container):
                raise TypeError(
                    "Number of output arguments provided ({}) is not as expected ({})".format(
                        len(out_container), len(output_types)
                    )
                )

            # Attach types recursively, if possible
            if not isinstance(out_objects, tuple) and not isinstance(out_objects, list):
                try:
                    out_objects.neural_type = out_types_list[0][1]
                except Exception:
                    pass

                # Perform output ndim check
                if hasattr(out_objects, 'shape'):
                    value_shape = out_objects.shape
                    type_shape = out_types_list[0][1].axes
                    name = out_types_list[0][0]

                    if type_shape is not None and len(value_shape) != len(type_shape):
                        raise TypeError(
                            f"Output shape mismatch occured for {name} in module {self.__class__.__name__} : \n"
                            f"Output shape expected = {type_shape} | \n"
                            f"Output shape found : {value_shape}"
                        )
            else:
                for ind, res in enumerate(out_objects):
                    self.__attach_neural_type(res, out_types_list[ind][1], name=out_types_list[ind][0])

    def __check_neural_type(self, obj, type_val, name=None):
        if isinstance(obj, tuple) or isinstance(obj, list):
            for elem in obj:
                self.__check_neural_type(elem, type_val, name=name)
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

        # Perform input ndim check
        if hasattr(obj, 'shape'):
            value_shape = obj.shape
            type_shape = type_val.axes

            if type_shape is not None and len(value_shape) != len(type_shape):
                raise TypeError(
                    f"Input shape mismatch occured for {name} in module {self.__class__.__name__} : \n"
                    f"Input shape expected = {type_shape} | \n"
                    f"Input shape found : {value_shape}"
                )

    def __attach_neural_type(self, obj, type_val, name=None):
        if isinstance(obj, tuple) or isinstance(obj, list):
            for elem in obj:
                self.__attach_neural_type(elem, type_val, name=name)
            return  # after processing nest, return to avoid argument insertion into nest itself

        try:
            obj.neural_type = type_val
        except Exception:
            pass

        # Perform output ndim check
        if hasattr(obj, 'shape'):
            value_shape = obj.shape
            type_shape = type_val.axes

            if type_shape is not None and len(value_shape) != len(type_shape):
                raise TypeError(
                    f"Output shape mismatch occured for {name} in module {self.__class__.__name__} : \n"
                    f"Output shape expected = {type_shape} | \n"
                    f"Output shape found : {value_shape}"
                )


class Serialization(ABC):
    @classmethod
    def from_config_dict(cls, config: DictConfig):
        """Instantiates object using DictConfig-based configuration"""
        # Resolve the config dict
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
            config = OmegaConf.create(config)
            OmegaConf.set_struct(config, True)

        if ('cls' in config or 'target' in config) and 'params' in config:
            # regular hydra-based instantiation
            instance = hydra.utils.instantiate(config=config)
        elif '_target_' in config:
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
            # Resolve the config dict
            config = OmegaConf.to_container(self._cfg, resolve=True)
            config = OmegaConf.create(config)
            OmegaConf.set_struct(config, True)

            self._cfg = config

            return self._cfg
        else:
            raise NotImplementedError(
                'to_config_dict() can currently only return object._cfg but current object does not have it.'
            )


class FileIO(ABC):
    def save_to(self, save_path: str):
        """Saves module/model with weights"""
        raise NotImplementedError()

    @classmethod
    def restore_from(
        cls,
        restore_path: str,
        override_config_path: Optional[str] = None,
        map_location: Optional['torch.device'] = None,
    ):
        """Restores module/model with weights"""
        raise NotImplementedError()

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


@dataclass
class PretrainedModelInfo:
    pretrained_model_name: str
    description: str
    location: str
    class_: 'Model' = None


class Model(Typing, Serialization, FileIO):
    """
    Abstract class offering interface which should be implemented by all NeMo models.
    """

    @classmethod
    @abstractmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        Should list all pre-trained models available via NVIDIA NGC cloud

        Returns:
            A list of PretrainedModelInfo entries
        """
        pass

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        refresh_cache: bool = False,
        override_config_path: Optional[str] = None,
        map_location: Optional['torch.device'] = None,
    ):
        """
        Instantiates an instance of NeMo from NVIDIA NGC cloud
        Use restore_from() to instantiate from a local .nemo file.
        Args:
            model_name: string key which will be used to find the module.
            refresh_cache: If set to True, then when fetching from cloud, this will re-fetch the file
                from cloud even if it is already found in a cache locally.
            override_config_path: path to a yaml config that will override the internal
                config file
            map_location: Optional torch.device() to map the instantiated model to a device.
                By default (None), it will select a GPU if available, falling back to CPU otherwise.
        Returns:
            A model instance of a particular model class
        """
        location_in_the_cloud = None
        description = None
        if cls.list_available_models() is not None:
            for pretrained_model_info in cls.list_available_models():
                if pretrained_model_info.pretrained_model_name == model_name:
                    location_in_the_cloud = pretrained_model_info.location
                    description = pretrained_model_info.description
                    class_ = pretrained_model_info.class_
        if location_in_the_cloud is None:
            raise FileNotFoundError(
                f"Model {model_name} was not found. Check cls.list_available_models() for the list of all available models."
            )
        filename = location_in_the_cloud.split("/")[-1]
        url = location_in_the_cloud.replace(filename, "")
        cache_dir = Path.joinpath(Path.home(), f'.cache/torch/NeMo/NeMo_{nemo.__version__}/{filename[:-5]}')
        # If either description and location in the cloud changes, this will force re-download
        cache_subfolder = hashlib.md5((location_in_the_cloud + description).encode('utf-8')).hexdigest()
        # if file exists on cache_folder/subfolder, it will be re-used, unless refresh_cache is True
        nemo_model_file_in_cache = maybe_download_from_cloud(
            url=url, filename=filename, cache_dir=cache_dir, subfolder=cache_subfolder, refresh_cache=refresh_cache
        )
        logging.info("Instantiating model from pre-trained checkpoint")
        if class_ is None:
            class_ = cls
        instance = class_.restore_from(
            restore_path=nemo_model_file_in_cache, override_config_path=override_config_path, map_location=map_location
        )
        return instance


class typecheck:
    class TypeState(Enum):
        """
        Placeholder to denote the default value of type information provided.
        If the constructor of this decorator is used to override the class level type definition,
        this enum value indicate that types will be overridden.
        """

        UNINITIALIZED = 0

    def __init__(
        self,
        input_types: Union[TypeState, Dict[str, NeuralType]] = TypeState.UNINITIALIZED,
        output_types: Union[TypeState, Dict[str, NeuralType]] = TypeState.UNINITIALIZED,
    ):
        """
        A decorator which performs input-output neural type checks, and attaches
        neural types to the output of the function that it wraps.

        Requires that the class inherit from `nemo.core.Typing` in order to perform
        type checking, and will raise an error if that is not the case.

        # Usage (Class level type support)
        @typecheck()
        def fn(self, arg1, arg2, ...):
            ...

        # Usage (Function level type support)
        @typecheck(input_types=..., output_types=...)
        def fn(self, arg1, arg2, ...):
            ...

        Points to be noted:
        1) The brackets () in `@typecheck()` are necessary.

            You will encounter a TypeError: __init__() takes 1 positional argument but X
            were given without those brackets.

        2) The function can take any number of positional arguments during definition.

            When you call this function, all arguments must be passed using kwargs only.

        """
        self.input_types = input_types
        self.output_types = output_types

        if input_types == self.TypeState.UNINITIALIZED:
            self.input_override = False
        else:
            self.input_override = True

        if output_types == self.TypeState.UNINITIALIZED:
            self.output_override = False
        else:
            self.output_override = True

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

        # Preserve type information
        if self.input_types is typecheck.TypeState.UNINITIALIZED:
            self.input_types = instance.input_types

        if self.output_types is typecheck.TypeState.UNINITIALIZED:
            self.output_types = instance.output_types

        # Resolve global type or local overridden type
        if self.input_override:
            input_types = self.input_types
        else:
            input_types = instance.input_types

        if self.output_override:
            output_types = self.output_types
        else:
            output_types = instance.output_types

        # If types are not defined, skip type checks and just call the wrapped method
        if input_types is None and output_types is None:
            return wrapped(*args, **kwargs)

        # Check that all arguments are kwargs
        if input_types is not None and len(args) > 0:
            raise TypeError("All arguments must be passed by kwargs only for typed methods")

        # Perform rudimentary input checks here
        instance._validate_input_types(input_types=input_types, **kwargs)

        # Call the method - this can be forward, or any other callable method
        outputs = wrapped(*args, **kwargs)

        instance._attach_and_validate_output_types(output_types=output_types, out_objects=outputs)

        return outputs

    @staticmethod
    def set_typecheck_enabled(enabled: bool = True):
        global _TYPECHECK_ENABLED
        _TYPECHECK_ENABLED = enabled

    @staticmethod
    @contextmanager
    def disable_checks():
        typecheck.set_typecheck_enabled(enabled=False)
        try:
            yield
        finally:
            typecheck.set_typecheck_enabled(enabled=True)
