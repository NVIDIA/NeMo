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

__all__ = ['WeightShareTransform', 'NeuralModule', 'PretrainedModelInfo', 'ModuleType', 'OperationMode']

import uuid
from abc import abstractmethod
from collections import namedtuple
from enum import Enum
from inspect import getargvalues, getfullargspec, stack
from os import path
from typing import Any, Dict, List, Optional, Set, Tuple

from ruamel.yaml import YAML

from nemo.core.neural_factory import NeuralModuleFactory, OperationMode
from nemo.core.neural_interface import NeuralInterface
from nemo.core.neural_types import NeuralPortNameMismatchError, NeuralType, NmTensor
from nemo.package_info import __version__ as nemo_version
from nemo.utils import logging
from nemo.utils.decorators.deprecated import deprecated
from nemo.utils.neural_graph.connection import StepModulePort

YAML = YAML(typ='safe')


class ModuleType(Enum):
    """ Back-end independent module types """

    module = 0
    datalayer = 1
    trainable = 2
    loss = 3
    nontrainable = 4


class WeightShareTransform(Enum):
    """When sharing parameters, what kind of transform to apply."""

    SAME = 0
    TRANSPOSE = 1


PretrainedModelInfo = namedtuple(
    "PretrainedModleInfo", ("pretrained_model_name", "description", "parameters", "location"),
)


class NeuralModule(NeuralInterface):
    """
    Abstract class that every Neural Module must inherit from.
    """

    def __init__(self, name=None):
        # Initialize the inferface.
        super().__init__()

        # Retrieve dictionary of parameters (keys, values) passed to init.
        self._init_params = self.__extract_init_params()

        # Get object UUID.
        self._uuid = str(uuid.uuid4())

        # Register module and store the generated name.
        self._name = self._app_state.register_module(self, name)

        # Set "module" type as default.
        self._type = ModuleType.module

        # Set "both" as default operation mode.
        self._operation_mode = OperationMode.both

        # Get default factory.
        self._factory = NeuralModuleFactory.get_default_factory()

        # Set module properties from factory else use defaults
        self._placement = self._factory.placement
        # If one needs to change that should override it manually.

        # Optimization level.
        self._opt_level = self._factory.optim_level

    @property
    def init_params(self) -> Dict[str, Any]:
        """
        Property returning parameters used to instantiate the module.

        Returns:
            Dictionary containing parameters used to instantiate the module.
        """
        return self._init_params

    def __extract_init_params(self) -> Dict[str, Any]:
        """
        Retrieves the dictionary of of parameters (keys, values) passed to constructor of a class derived
        (also indirectly) from the Neural Module class.

        Returns:
            Dictionary containing parameters passed to init().
        """
        # Get names of arguments of the original module init method.
        to_set_params = getfullargspec(type(self).__init__).args
        to_set_params.remove("self")

        # Create empty list of init params.
        init_params = {}

        # Get the frame "call context".
        for frame in stack()[1:]:
            # Get the current call arguments.
            localvars = getargvalues(frame[0])

            # Fill the parameters with call arguments.
            for key in to_set_params:
                if key in localvars.args:
                    init_params[key] = localvars.locals[key]

            # Remove all set keys.
            for key in init_params.keys():
                if key in to_set_params:
                    to_set_params.remove(key)

            # Check if we have set everything.
            if len(to_set_params) == 0:
                break

        # Make sure that we collected ALL (and ONLY) the signature params - if not, then there is a BUG!
        if len(to_set_params) != 0:
            raise ValueError(
                "Could not collect all the signature params! "
                f"Please file a bug on GitHub with the current stack trace so that it can be reproduced."
            )

        # print("! init_params of {}: {}\n".format(type(self).__name__, init_params))

        # Return parameters.
        return init_params

    def __validate_params(self, params: Dict[str, Any]) -> bool:
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
                logging.warning(
                    "Parameter '{}' contains a variable '{}' of type '{}' which is not allowed.".format(
                        key, variable, type(variable)
                    )
                )
                ok = False

        # Return the result.
        return ok

    def __is_of_allowed_type(self, var) -> bool:
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

    def export_to_config(self, config_file: str):
        """
        A function that exports module "configuration" (i.e. init parameters) to a YAML file.

        Args:
            config_file: path (absolute or relative) and name of the config file (YML)
        Raises:
            ValueError: An error occurred and  parameters coudn't be exported.
        """
        # Greate an absolute path.
        abs_path_file = path.expanduser(config_file)

        # Serialize the module.
        to_export = self.serialize()

        # All parameters are ok, let's export.
        with open(abs_path_file, 'w') as outfile:
            YAML.dump(to_export, outfile)

        logging.info(
            "Configuration of module `{}` ({}) exported to '{}'".format(self.name, type(self).__name__, abs_path_file)
        )

    def serialize(self) -> Dict[str, Any]:
        """
        A method serializing the whole Neural module (into a dictionary).

        Returns:
            Dictionary containing a "serialized" module.
        """
        # Create a dictionary representing the serialized object.
        serialized_module = {}

        # Add "header" with module "specification".
        serialized_module["header"] = self.__serialize_header()

        # Add init parameters.
        serialized_module["init_params"] = self._serialize_configuration()

        # Return the dictionary.
        return serialized_module

    def __serialize_header(self) -> Dict[str, Any]:
        """
        A protected method that creates a header stored later in the configuration file.
            
        Returns:
            Dictionary containing a header with module specification.
        """

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

        # Create a "header" with module "specification".
        header = {
            "nemo_core_version": nemo_version,
            "collection_type": collection_type,
            "collection_version": collection_version,
            # "class": module_class_name, # Operating only on full_spec now.
            "full_spec": module_full_spec,
        }
        return header

    def _serialize_configuration(self) -> Dict[str, Any]:
        """
        A function that serializes the module "configuration (i.e. init parameters) to a dictionary.

        ..note:
            Thus functions should be overloaded when writing a custom module import/export.

        Returns:
            A "serialized" dictionary with module configuration.
        Raises:
            A ValueError exception in case then parameters coudn't be exported.
        """
        # Check if generic export will work.
        if not self.__validate_params(self._init_params):
            raise ValueError(
                "Generic configuration export enables to use of parameters of primitive types (string, int, float) "
                F"or (lists of/dicts of) primitive types. Please implement your own custom `export_to_config()` and "
                F"`import_from_config()` methods for your custom Module class."
            )
        # In this case configuration = init parameters.
        return self._init_params

    @classmethod
    def import_from_config(
        cls, config_file: str, section_name: str = None, name: str = None, overwrite_params: Dict = {}
    ) -> 'NeuralModule':
        """
        Class method importing the configuration file.
        Raises an ImportError exception when config file is invalid or
        incompatible (when called from a particular class).

        Args:
            config_file: path (absolute or relative) and name of the config file (YML)
            section_name: section in the configuration file storing module configuration (optional, DEFAULT: None)
            name: name of the module that will overwrite the name in the `init_params` (optional, DEFAULT: None)
            overwrite_params: Dictionary containing parameters that will be added to or overwrite (!)
            the default init parameters loaded from the configuration file (the module "init_params" section).
        Returns:
            Instance of the created NeuralModule object.
        """
        logging.info("Loading configuration of a new Neural Module from the `{}` file".format(config_file))

        # Validate the content of the configuration file (its header).
        loaded_config = cls.__validate_config_file(config_file, section_name)

        # "Deserialize" the module.
        obj = cls.deserialize(loaded_config, name, overwrite_params)

        # Return the new module.
        return obj

    @classmethod
    def __validate_config_file(cls, config_file: str, section_name: str = None) -> Dict[str, Any]:
        """
        Class method validating whether the config file has a proper content (sections, specification etc.).
        Raises an ImportError exception when config file is invalid or
        incompatible (when called from a particular class).

        Args:
            config_file: path (absolute or relative) and name of the config file (YML)
            section_name: section in the configuration file storing module configuration (optional, DEFAULT: None)
        Returns:
            A loaded configuration file (dictionary).
        """
        # Greate an absolute path.
        abs_path_file = path.expanduser(config_file)

        # Open the config file.
        with open(abs_path_file, 'r') as stream:
            loaded_config = YAML.load(stream)

        # Check section.
        if section_name is not None:
            if section_name not in loaded_config:
                raise ImportError(
                    "The loaded config `{}` doesn't contain the indicated `{}` section".format(
                        config_file, section_name
                    )
                )
            # Section exists - use only it for configuration.
            loaded_config = loaded_config[section_name]

        # Make sure that the config is valid.
        if "header" not in loaded_config:
            raise ImportError("The loaded config `{}` doesn't contain the `header` section".format(config_file))

        if "init_params" not in loaded_config:
            raise ImportError("The loaded config `{}` doesn't contain the `init_params` section".format(config_file))

        # Parse the "full specification".
        spec_list = loaded_config["header"]["full_spec"].split(".")

        # Check if config contains data of a compatible class.
        if not issubclass(cls.__deserialize_header(loaded_config["header"]), cls):
            txt = "The loaded file `{}` contains configuration of ".format(config_file)
            txt = txt + "`{}` thus cannot be used for instantiation of an object of type `{}`".format(
                spec_list[-1], cls.__name__
            )
            raise ImportError(txt)

        # Success - return configuration.
        return loaded_config

    @classmethod
    def deserialize(
        cls, configuration: Dict[str, Any], name: str = None, overwrite_params: Dict[str, Any] = {}
    ) -> 'NeuralModule':
        """
        Class method instantiating the neural module object based on the configuration (dictionary).

        Args:
            configuration: Dictionary containing proper "header" and "init_params" sections.

            name: name of the module that will overwrite the name in the `init_params` (optional, DEFAULT: None)

            overwrite_params: Dictionary containing parameters that will be added to or overwrite (!)
            the default init parameters loaded from the configuration file (the module "init_params" section).

        Returns:
            Instance of the created NeuralModule object.
        """
        # Deserialize header - get object class.
        module_class = cls.__deserialize_header(configuration["header"])

        # Update parameters with additional ones.
        configuration["init_params"].update(overwrite_params)

        # Override module name in init_params using the logic:
        #  * section_name if not none overrides init_params.name first (skipped for now, TOTHINK!)
        #  * name (if None) overrides init_params.name
        if name is not None:
            configuration["init_params"]["name"] = name

        # Get init parameters.
        init_params = cls._deserialize_configuration(configuration["init_params"])

        # Create the module instance.
        new_module = module_class(**init_params)
        logging.info(
            "Instantiated a new Neural Module named `{}` of type `{}`".format(
                new_module.name, type(new_module).__name__
            )
        )

        # Return the module instance.
        return new_module

    @classmethod
    def __deserialize_header(cls, serialized_header: Dict[str, Any]):
        """
        Method deserializes the header and extracts the module class.

        Args:
            serialized_header: Dictionary containing module header.
        Returns:
            Class of the module to be created.
        """
        # Parse the "full specification".
        spec_list = serialized_header["full_spec"].split(".")

        # Get module class from the "full specification".
        mod_obj = __import__(spec_list[0])
        for spec in spec_list[1:]:
            mod_obj = getattr(mod_obj, spec)

        # Return "class".
        return mod_obj

    @classmethod
    def _deserialize_configuration(cls, serialized_init_params: Dict[str, Any]):
        """
        A function that deserializes the module "configuration (i.e. init parameters).

        ..note:
            Thus functions should be overloaded when writing a custom module import/export.

        Args:
            serialized_init_params: List of init parameters loaded from the file.
        Returns:
            A "deserialized" list with init parameters.
        """
        # In this case configuration = init parameters.
        return serialized_init_params

    @property
    @abstractmethod
    def input_ports(self) -> Dict[str, NeuralType]:
        """
        Returns definitions of module input ports

        Returns:
          A dictionary containing module's input ports (names, NeuralTypes) mapping.
        """

    @property
    @abstractmethod
    def output_ports(self) -> Dict[str, NeuralType]:
        """
        Returns definitions of module output ports

        Returns:
          A dictionary containing module's output ports (names, NeuralTypes) mapping.
        """

    @property
    def _disabled_deployment_input_ports(self) -> Set[str]:
        """Returns names of input ports that will not be included in an export

        Returns:
          A (set) of module's input port names that are not exportable
        """
        return set([])

    @property
    def _disabled_deployment_output_ports(self) -> Set[str]:
        """Returns names of output ports that will not be included in an export

        Returns:
          A (set) of module's output port names that are not exportable
        """
        return set([])

    def _prepare_for_deployment(self) -> None:
        """Patch the module if required to prepare for deployment

        Returns:
            (Optional) input and output example tensors
        """
        return None, None

    @property
    def operation_mode(self):
        """ Returns the operation mode. """
        return self._operation_mode

    @property
    def type(self):
        """ Returns the type of module. """
        return self._type

    @operation_mode.setter
    def operation_mode(self, operation_mode: OperationMode):
        """ Sets the operation mode. """
        self._operation_mode = operation_mode

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
        # print(" Neural Module:__call__")

        # Set the operation mode of the outer graph.
        self.operation_mode = self._app_state.active_graph.operation_mode
        # The input and output ports definitions can potentially depend on the operation mode!

        # Record the operation (i.e. add a single module).
        step_number = self._app_state.active_graph.record_step(self)

        ###### PROCESS INPUTS. ######
        # Iterate through all passed parameters.
        for port_name, port_content in kwargs.items():
            # Make sure that passed arguments corresponds to one of the input port names.
            if port_name not in self.input_ports.keys():
                raise NeuralPortNameMismatchError(port_name)

            # At that point the input can be one of three types:
            # * NeuralGraph -> bind port using the default name and type.
            # * GraphInput -> check definition, if ok bind port.
            # * NmTensor -> check definition, add self as a "consumer" of a tensor (produced by other module).

            # Check what was actually passed.
            if type(port_content).__name__ == "NeuralGraph":
                # Make sure that port_content is the currently active graph!
                if port_content is not self._app_state.active_graph:
                    raise ConnectionError("Ports can be bound only by passing the active graph object!")
                # Create an alias so the logic will be more clear.
                active_graph = port_content

                # This case: we are nesting one graph into another and must bind input port of one graph in another!
                # So generally we must "copy" the of thus module to graog (the inverted logic!).

                # Copy the port "definition" (i.e. is NeuralType) using the same port name.
                active_graph.inputs[port_name] = self.input_ports[port_name]

                # Bind the neural graph input port, i.e. remember that a given graph port should pass data
                # to THIS module-port (when it finally will be connected).
                active_graph.inputs[port_name].bind(StepModulePort(step_number, self.name, port_name))

                # Please note that there are no "consumers" here - this is a "pure binding".

            elif type(port_content).__name__ == "GraphInput":

                # Check if GraphInput belongs to the active graph !
                own_port = False
                for gcontent in self._app_state.active_graph.inputs.values():
                    if gcontent is port_content:
                        own_port = True
                        break
                if not own_port:
                    raise NeuralPortNameMismatchError(port_name)

                # Compare input port definition with the received definition.
                self.input_ports[port_name].compare_and_raise_error(
                    self.__class__.__name__, port_name, port_content.ntype
                )

                # Bind the neural graph input port, i.e. remember that a given graph port should pass data
                # to THIS module-port (when it finally will be connected).
                port_content.bind(StepModulePort(step_number, self.name, port_name))

                # Please note that there are no "consumers" here - this is a "pure binding".

            elif type(port_content) is NmTensor:
                # Compare input port definition with the received definition.
                self.input_ports[port_name].compare_and_raise_error(self.__class__.__name__, port_name, port_content)

                # Ok, the goal here is to actually "connect": add self (module) as "consumer" to the input tensor.
                port_content.add_consumer(StepModulePort(step_number, self.name, port_name))
            else:
                raise TypeError(
                    "Input '{}' must be of one of three types: NeuralGraph, GraphInput or NmTensor".format(port_name)
                )

        ###### PRODUCE OUTPUTS. ######
        output_port_defs = self.output_ports
        # Create output tensors.
        if len(output_port_defs) == 1:
            # Get port name and type.
            out_name = list(output_port_defs)[0]
            out_type = output_port_defs[out_name]

            # Create a single returned tensor.
            results = NmTensor(producer=self, producer_args=kwargs, output_port_name=out_name, ntype=out_type,)

            # Bind the "default" output ports.
            self._app_state.active_graph.bind_outputs(results)
        else:
            # Create output tensors.
            output_tensors = []
            for out_name, out_type in output_port_defs.items():
                output_tensors.append(
                    NmTensor(producer=self, producer_args=kwargs, output_port_name=out_name, ntype=out_type,)
                )

            # Create a named tuple type enabling to access outputs by attributes (e.g. out.x).
            output_class_name = f'{self.__class__.__name__}Output'
            result_type = namedtuple(typename=output_class_name, field_names=output_port_defs.keys())

            # Create the returned tuple object.
            results = result_type(*output_tensors)

            # Bind the output tensors.
            self._app_state.active_graph.bind_outputs(output_tensors)

        # Return the results.
        return results

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
