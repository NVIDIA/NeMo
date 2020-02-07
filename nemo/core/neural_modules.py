# Copyright (c) 2019 NVIDIA Corporation
"""This file contains NeuralModule and NmTensor classes."""
__all__ = ['WeightShareTransform', 'NeuralModule']

import collections
import uuid
from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum
from inspect import getargvalues, getfullargspec, stack
from os import path
from typing import Dict, List, Optional, Set, Tuple

import yaml

import nemo
from .neural_types import (
    CanNotInferResultNeuralType,
    NeuralPortNameMismatchError,
    NeuralPortNmTensorMismatchError,
    NeuralType,
    NeuralTypeComparisonResult,
    NmTensor,
)
from nemo import logging
from nemo.core import NeuralModuleFactory
from nemo.package_info import __version__ as nemo_version
from nemo.utils.decorators.deprecated import deprecated


class WeightShareTransform(Enum):
    """When sharing parameters, what kind of transform to apply."""

    SAME = 0
    TRANSPOSE = 1


PretrainedModelInfo = namedtuple(
    "PretrainedModleInfo", ("pretrained_model_name", "description", "parameters", "location"),
)


class NeuralModule(ABC):
    """Abstract class that every Neural Module must inherit from.
    """

    def __init__(self):

        # Get default factory.
        self._factory = NeuralModuleFactory.get_default_factory()

        # Set module properties from factory else use defaults
        self._placement = self._factory.placement
        # If one needs to change that should override it manually.

        # Optimization level.
        self._opt_level = self._factory.optim_level

        # Get object UUID.
        self._uuid = str(uuid.uuid4())

        # Retrieve dictionary of parameters (keys, values) passed to init.
        self._init_params = self.__extract_init_params()

        # Pint the types of the values.
        # for key, value in self._init_params.items():
        #    print("{}: {} ({})".format(key, value, type(value)))

        # Validate the parameters.
        # self._validate_params(self._init_params)

    @property
    def init_params(self) -> Optional[Dict]:
        """
            Property returning parameters used to instantiate the module.

            Returns:
                Dictionary containing parameters used to instantiate the module.
        """
        return self._init_params

    def __extract_init_params(self):
        """
            Retrieves the dictionary of of parameters (keys, values) passed to constructor of a class derived
            (also indirectly) from the Neural Module class.

            Returns:
                Dictionary containing parameters passed to init().
        """
        # Get names of arguments of the original module init method.
        init_keys = getfullargspec(type(self).__init__).args

        # Remove self.
        if "self" in init_keys:
            init_keys.remove("self")

        # Create list of params.
        init_params = {}.fromkeys(init_keys)

        # Retrieve values of those params from the call list.
        for frame in stack()[1:]:
            localvars = getargvalues(frame[0]).locals
            # print("localvars: ", localvars)
            for key in init_keys:
                # Found the variable!
                if key in localvars.keys():
                    # Save the value.
                    init_params[key] = localvars[key]

        # Return parameters.
        return init_params

    def __validate_params(self, params):
        """
            Checks whether dictionary contains parameters being primitive types (string, int, float etc.)
            or (lists of)+ primitive types.

            Args:
                params: dictionary of parameters.

            Returns:
                True if all parameters were ok, False otherwise.
        """
        ok = True

        # Iterate over parameters and check them one by one.
        for key, variable in params.items():
            if not self.__is_of_allowed_type(variable):
                nemo.logging.warning(
                    "Parameter '{}' contains a variable '{}' of type '{}' which is not allowed.".format(
                        key, variable, type(variable)
                    )
                )
                ok = False

        # Return the result.
        return ok

    def __is_of_allowed_type(self, var):
        """
            A recursive function that checks if a given variable is of allowed type.

            Args:
                pretrained_model_name (str): name of pretrained model to use in order.

            Returns:
                True if all parameters were ok, False otherwise.
        """
        # Special case: None is also allowed.
        if var is None:
            return True

        var_type = type(var)

        # If this is list - check its elements.
        if var_type == list:
            for list_var in var:
                if not self.__is_of_allowed_type(list_var):
                    return False

        # If this is dict - check its elements.
        elif var_type == dict:
            for _, dict_var in var.items():
                if not self.__is_of_allowed_type(dict_var):
                    return False

        elif var_type not in (str, int, float, bool):
            return False

        # Well, seems that everything is ok.
        return True

    def export_config(self, config_file, config_dir="~/data/configs"):
        """
            A function that exports module "configuration" (i.e. init parameters) to a YAML file.
            Raises a ValueError exception in case then parameters coudn't be exported.

            Args:
                config_file: yml file name

                config_dir: directory where the file will be stored (DEFAULT: ~/data/configs)
        """
        # Check if generic export will work.
        if not self.__validate_params(self._init_params):
            raise ValueError(
                "Generic Module export cannot work as some of the values are not primitive types (string, int, float) "
                F"or (lists of/dicts of) primitive types. Please implement your own custom `export_config()` and "
                F"`import_config()` methods for your custom Module class."
            )

        # Greate an absolute path.
        abs_path_file = path.join(path.expanduser(config_dir), config_file)

        # Create the dictionary to be exported.
        to_export = {}

        # Get module "full specification".
        module_full_spec = str(self.__module__) + "." + str(self.__class__.__qualname__)
        module_class_name = type(self).__name__
        # print(module_full_spec)

        # Check whether module belongs to a collection.
        spec_list = module_full_spec.split(".")

        # Do not check Neural Modules from unit tests.
        if spec_list[0] == "tests":
            # Set collection variables.
            collection_type = "tests"
            collection_version = None
        else:
            # Check if component belongs to any collection
            if len(spec_list) < 3 or (spec_list[0] != "nemo" and spec_list[1] != "collection"):
                logging.warning(
                    "Module `{}` does not belong to any collection. This won't be allowed in the next release.".format(
                        module_class_name
                    )
                )
                collection_type = "unknown"
                collection_version = None
            else:
                # Ok, set collection.
                collection_type = spec_list[2]
                collection_version = None
                # TODO: to be SET!
                # print(getattr("nemo.collections.nlp", __version__))

        # Add "header" with module "specification".
        to_export["header"] = {
            "nemo_version": nemo_version,
            "collection_type": collection_type,
            "collection_version": collection_version,
            "class": module_class_name,
            "full_spec": module_full_spec,
        }
        # Add init parameters.
        to_export["init_params"] = self._init_params
        # print(to_export)

        # All parameters are ok, let's export.
        with open(abs_path_file, 'w') as outfile:
            yaml.dump(to_export, outfile, default_flow_style=False)

        logging.info(
            "Configuration of module {} ({}) exported to {}".format(self._uuid, type(self).__name__, abs_path_file)
        )

    @classmethod
    def import_config(cls, config_file, config_dir="~/data/configs"):
        """
            Class method importing the configuration file.
            Raises an ImportError exception when config file is invalid or
            incompatible (when called from a particular class).

            Args:
                config_file: yml file name

                config_dir: directory where the file will be stored (DEFAULT: ~/data/configs)

            Returns:
                Instance of the created NeuralModule object.
        """

        # Greate an absolute path.
        abs_path_file = path.join(path.expanduser(config_dir), config_file)
        print(abs_path_file)

        # Open the config file.
        with open(abs_path_file, 'r') as stream:
            loaded_config = yaml.safe_load(stream)

        # Make sure that the config is valid.
        if "header" not in loaded_config:
            raise ImportError(
                "The loaded config `{}` from `{}` doesn't contain the`header` section".format(config_file, config_dir)
            )

        if "init_params" not in loaded_config:
            raise ImportError(
                "The loaded config `{}` from `{}` doesn't contain the`header` init_params".format(
                    config_file, config_dir
                )
            )

        # Check if config contains data of a compatible class.
        if cls.__name__ != "NeuralModule" and loaded_config["header"]["class"] != cls.__name__:
            txt = "The loaded file `{}` from `{}` contains configuration of ".format(config_file, config_dir)
            txt = txt + "`{}` thus cannot be used for instantiation of an object of type `{}`".format(
                loaded_config["header"]["class"], cls.__name__
            )
            raise ImportError(txt)

        # Get object class from "full specification".
        spec_list = loaded_config["header"]["full_spec"].split(".")
        mod_obj = __import__(spec_list[0])
        for spec in spec_list[1:]:
            mod_obj = getattr(mod_obj, spec)
        # print(mod_obj)

        # Get init parameters.
        init_params = loaded_config["init_params"]
        # Create and return the object.
        obj = mod_obj(**init_params)
        logging.info(
            "Instantiated a new Neural Module of type `{}` using configuration loaded from the `{}` file".format(
                loaded_config["header"]["class"], abs_path_file
            )
        )
        return obj

    @deprecated(version=0.11)
    @staticmethod
    def create_ports(**kwargs):
        """ Deprecated method, to be remoted in the next release."""
        raise Exception(
            'Deprecated method. Please implement ``inputs`` and ``outputs`` \
                 properties to define module ports instead'
        )

    @property
    @abstractmethod
    def input_ports(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module input ports

        Returns:
          A (dict) of module's input ports names to NeuralTypes mapping
        """

    @property
    @abstractmethod
    def output_ports(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports

        Returns:
          A (dict) of module's output ports names to NeuralTypes mapping
        """

    @staticmethod
    def pretrained_storage():
        return ''

    def __call__(self, **kwargs):
        """This method allows objects to be called with their port names

        Args:
          kwargs: Input ports and their values. For example:
          ...
          mymodule1 = Subclass1_of_NeuralModule(...)
          mymodule2 = Subclass2_of_NeuralModule(...)
          ...
          out_port1, out_port2 = mymodule1(input_port1=value1,
          input_port2=value2,
          input_port3=value3)
          out_port11 = mymodule2(input_port1=out_port2)
          ...

        Returns:
          NmTensor object or tuple of NmTensor objects
        """
        # if self._assigned_top_order is not None:
        #    raise ValueError("We currently do not support calling same NM"
        #                     "more than once")

        # Get input and output ports definitions.
        input_port_defs = self.input_ports
        output_port_defs = self.output_ports

        first_input_nmtensor_type = None
        input_nmtensors_are_of_same_type = True
        for port_name, tgv in kwargs.items():
            if port_name not in input_port_defs.keys():
                raise NeuralPortNameMismatchError("Wrong input port name: {0}".format(port_name))

            type_comatibility = input_port_defs[port_name].compare(tgv)

            if first_input_nmtensor_type is None:
                first_input_nmtensor_type = NeuralType(tgv._axis2type)
            else:
                if first_input_nmtensor_type._axis2type is None:
                    input_nmtensors_are_of_same_type = True
                else:
                    input_nmtensors_are_of_same_type = first_input_nmtensor_type.compare(
                        tgv
                    ) == NeuralTypeComparisonResult.SAME and len(first_input_nmtensor_type._axis2type)
            if not (
                type_comatibility == NeuralTypeComparisonResult.SAME
                or type_comatibility == NeuralTypeComparisonResult.GREATER
            ):
                raise NeuralPortNmTensorMismatchError(
                    "\n\nIn {0}. \n"
                    "Port: {1} and a NmTensor it was fed are \n"
                    "of incompatible neural types:\n\n{2} \n\n and \n\n{3}"
                    "\n\nType comparison result: {4}".format(
                        self.__class__.__name__, port_name, input_port_defs[port_name], tgv, type_comatibility,
                    )
                )
            if type_comatibility == NeuralTypeComparisonResult.LESS:
                logging.info('Types were raised')

        if len(output_port_defs) == 1:
            out_name = list(output_port_defs)[0]
            out_type = output_port_defs[out_name]
            if out_type is None:
                if input_nmtensors_are_of_same_type:
                    out_type = first_input_nmtensor_type
                else:
                    raise CanNotInferResultNeuralType(
                        "Can't infer output neural type." "Likely your inputs are of " "different type."
                    )
            return NmTensor(producer=self, producer_args=kwargs, name=out_name, ntype=out_type,)
        else:
            result = []
            for out_port, n_type in output_port_defs.items():
                out_type = n_type
                if out_type is None:
                    if input_nmtensors_are_of_same_type:
                        out_type = first_input_nmtensor_type
                    else:
                        raise CanNotInferResultNeuralType(
                            "Can't infer output neural type." "Likely your inputs are of " "different type."
                        )
                result.append(NmTensor(producer=self, producer_args=kwargs, name=out_port, ntype=out_type,))

            # Creating ad-hoc class for returning from module's forward pass.
            output_class_name = f'{self.__class__.__name__}Output'
            field_names = list(output_port_defs)
            result_type = collections.namedtuple(typename=output_class_name, field_names=field_names,)

            # Tie tuple of output tensors with corresponding names.
            result = result_type(*result)

            return result

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def get_weights(self) -> Optional[Dict[(str, bool)]]:
        """Returns NeuralModule's weights copy.

        Returns:
          Dictionary of name -> (weights, trainable)"""
        pass

    @abstractmethod
    def set_weights(
        self,
        name2weight: Dict[(str, Tuple[str, bool])],
        name2name_and_transform: Dict[(str, Tuple[str, WeightShareTransform])] = None,
    ):
        """Sets weight from given values. For every named weight in
        name2weight,
        if weight with the same name is found in the model, it will be set to
        found value.

        WARNING: This will NOT tie weights. It will copy values.

        If ``name2name_and_transform`` is provided then if will set weights
        using
        name mapping and transform. For example, suppose ``objec1.X = 3x5
        weight``.
        Then, if ``name2name_and_transform['X']=('Y',
        WeightShareTransform.TRANSPOSE)``
        and ``Y`` is 5x3 weight and ``name2weight['Y']=Y. Then:
        ``object1.set_weights(name2weight, name2name_and_transform)`` will
        set object1.X=transpose(Y).

        Args:
          name2weight (dict): dictionary of name to (weight, trainable).
          Typically this is output of get_weights method.
          name2name_and_transform: mapping from name -> (name, transform)
        """
        pass

    @staticmethod
    def list_pretrained_models() -> Optional[List[PretrainedModelInfo]]:
        """List all available pre-trained models (e.g. weights) for this NM.

        Returns:
            A list of PretrainedModelInfo tuples.
            The pretrained_model_name field of the tuple can be used to
            retrieve pre-trained model's weights (pass it as
            pretrained_model_name argument to the module's constructor)
        """
        return None

    def get_config_dict_and_checkpoint(self, pretrained_model_name):
        """WARNING: This part is work in progress"""
        return None

    @abstractmethod
    def tie_weights_with(
        self,
        module,
        weight_names=List[str],
        name2name_and_transform: Dict[(str, Tuple[str, WeightShareTransform])] = None,
    ):
        """Ties weights between self and module. For every weight name in
        weight_names, if weight with the same name is found in self, it will
        be tied
        with a same weight from ``module``.

        WARNING: Once weights are tied, updates to one weights's weights
        will affect
        other module's weights.


        If ``name2name_and_transform`` is provided then if will set weights
        using
        name mapping and transform. For example, suppose ``objec1.X = 3x5
        weights``
        and ``object2.Y = 5x3 weights``. Then these weights can be tied like
        this:

        .. code-block:: python

          object1.tie_weights_with(object2, weight_names=['X'],
          name2name_and_transform =
          { 'X': ('Y', WeightShareTransform.TRANSPOSE)})


        Args:
            module: with which module to tie weights
            weight_names (List[str]): list of self weights' names
            name2name_and_transform: mapping from name -> (name, transform)
        """
        pass

    def is_trainable(self) -> bool:
        """
        Checks if NeuralModule is trainable.
        A NeuralModule is trainable IFF it contains at least one trainable
        weight

        Returns:
          True if module has trainable weights, False otherwise
        """
        weights = self.get_weights()
        if weights is None:
            return False
        for name, w in weights.items():
            if w[1]:
                return True
        return False

    @abstractmethod
    def save_to(self, path: str):
        """Save module state to file.

        Args:
          path (string): path to while where to save.
        """
        pass

    @abstractmethod
    def restore_from(self, path: str):
        """Restore module's state from file.

        Args:
          path (string): path to where to restore from.
        """
        pass

    @abstractmethod
    def freeze(self, weights: Set[str] = None):
        """Freeze weights

        Args:
          weights (set): set of weight names to freeze
          If None, all weights are freezed.
        """
        pass

    @abstractmethod
    def unfreeze(self, weights: Set[str] = None):
        """Unfreeze weights

        Args:
          weights (set): set of weight names to unfreeze
          If None, all weights are unfreezed.
        """
        pass

    @property
    def placement(self):
        """Module's placement. Currently CPU or GPU.
        DataParallel and ModelParallel will come later.

        Returns:
          (DeviceType) Device where NM's weights are located
        """
        return self._placement

    @property
    @deprecated(version=0.11)
    def local_parameters(self) -> Optional[Dict]:
        """Get module's parameters

        Returns:
          module's parameters
        """
        return self._init_params
        # return self._local_parameters

    @property
    def unique_instance_id(self):
        """A unique instance id for this object

        Returns:
          A uniq uuid which can be used to identify this object
        """
        return self._uuid

    @property
    def factory(self):
        """ Neural module factory which created this module
        Returns: NeuralModuleFactory instance or None
        """
        return self._factory

    @property
    @abstractmethod
    def num_weights(self):
        """Number of module's weights
        """
        pass
