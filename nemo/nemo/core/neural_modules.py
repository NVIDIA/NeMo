# Copyright (c) 2019 NVIDIA Corporation
"""This file contains NeuralModule and NmTensor classes."""
import uuid
import logging
from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum
from typing import Optional, Dict, Set, Tuple, List
from inspect import getargvalues, stack
from nemo.core import NeuralModuleFactory

from .neural_factory import Optimization, DeviceType
from .neural_types import CanNotInferResultNeuralType,\
                          NeuralType, NeuralTypeComparisonResult,\
                          NeuralPortNameMismatchError,\
                          NeuralPortNmTensorMismatchError,\
                          NmTensor


class WeightShareTransform(Enum):
    """When sharing parameters, what kind of transform to apply."""

    SAME = 0
    TRANSPOSE = 1


PretrainedModelInfo = namedtuple("PretrainedModleInfo",
                                 ("pretrained_model_name", "description",
                                  "parameters", "location"))


class NeuralModule(ABC):
    """Abstract class that every Neural Module must inherit from.

    Args:
        pretrained_model_name (str): name of pretrained model to use in order
            to initialize this neural module
        create_port_args (dict): arguments that are passed to create_ports()
        factory (NeuralModuleFactory): :class:`NeuralModuleFactory` which
            created or which should mange this instance. Required for
            multi-gpu training.
        placement (DeviceType): (default:None) where this module should
            be placed. If provided, this parameter takes precedence over
            whatever is specified in factory.
    """

    def __init__(
            self, *,
            pretrained_model_name=None,
            create_port_args=None,
            factory=None,
            placement=None,
            **kwargs
    ):
        self._pretrained_model_name = pretrained_model_name
        self._local_parameters = self.update_local_params()

        if create_port_args is None:
            create_port_args = {}
        self._input_ports, self._output_ports = self.create_ports(
            **create_port_args)

        default_factory = NeuralModuleFactory.get_default_factory()
        if (factory is None) and (default_factory is not None):
            factory = default_factory

        # Set module properties from factory else use defaults
        self._placement = factory.placement if factory is not None\
            else DeviceType.GPU
        self._opt_level = factory.optim_level if factory is not None\
            else Optimization.mxprO0
        self._logger = factory.logger if factory is not None\
            else logging

        # Update module properties using overrides if overrides exist
        if placement is not None:
            self._placement = placement

        self._factory = factory
        self._uuid = str(uuid.uuid4())

        # if kwargs:
        #    self._logger.warning(
        #        "When constructing {}. The base "
        #        "NeuralModule class received the following unused "
        #        "arguments:".format(self.__class__.__name__))
        #    self._logger.warning("{}".format(kwargs.keys()))

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

        first_input_nmtensor_type = None
        input_nmtensors_are_of_same_type = True
        for port_name, tgv in kwargs.items():
            if port_name not in self._input_ports.keys():
                raise NeuralPortNameMismatchError(
                    "Wrong input port name: {0}".format(port_name)
                )

            type_comatibility = self._input_ports[port_name].compare(tgv)

            if first_input_nmtensor_type is None:
                first_input_nmtensor_type = NeuralType(tgv._axis2type)
            else:
                if first_input_nmtensor_type._axis2type is None:
                    input_nmtensors_are_of_same_type = True
                else:
                    input_nmtensors_are_of_same_type = \
                        first_input_nmtensor_type.compare(tgv) \
                        == NeuralTypeComparisonResult.SAME and \
                        len(first_input_nmtensor_type._axis2type)
            if not (type_comatibility == NeuralTypeComparisonResult.SAME or
                    type_comatibility == NeuralTypeComparisonResult.GREATER):
                raise NeuralPortNmTensorMismatchError(
                    "\n\nIn {0}. \n"
                    "Port: {1} and a NmTensor it was fed are \n"
                    "of incompatible neural types:\n\n{2} \n\n and \n\n{3}"
                    "\n\nType comparison result: {4}"
                    .format(
                        self.__class__.__name__,
                        port_name,
                        self._input_ports[port_name],
                        tgv,
                        type_comatibility
                    )
                )
            if type_comatibility == NeuralTypeComparisonResult.LESS:
                print('Types were raised')

        if len(self._output_ports) == 1:
            out_name = list(self._output_ports)[0]
            out_type = self._output_ports[out_name]
            if out_type is None:
                if input_nmtensors_are_of_same_type:
                    out_type = first_input_nmtensor_type
                else:
                    raise CanNotInferResultNeuralType(
                        "Can't infer output neural type."
                        "Likely your inputs are of "
                        "different type."
                    )
            return NmTensor(
                producer=self, producer_args=kwargs, name=out_name,
                ntype=out_type
            )
        else:
            result = []
            for out_port, n_type in self._output_ports.items():
                out_type = n_type
                if out_type is None:
                    if input_nmtensors_are_of_same_type:
                        out_type = first_input_nmtensor_type
                    else:
                        raise CanNotInferResultNeuralType(
                            "Can't infer output neural type."
                            "Likely your inputs are of "
                            "different type."
                        )
                result.append(
                    NmTensor(
                        producer=self,
                        producer_args=kwargs,
                        name=out_port,
                        ntype=out_type,
                    )
                )
            return tuple(result)

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
            name2name_and_transform: Dict[
                (str, Tuple[str, WeightShareTransform])] = None,
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
            name2name_and_transform: Dict[
                (str, Tuple[str, WeightShareTransform])] = None,
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
    def local_parameters(self) -> Optional[Dict]:
        """Get module's parameters

        Returns:
          module's parameters
        """
        return self._local_parameters

    @property
    def input_ports(self) -> Optional[Dict[str, NeuralType]]:
        """Get all module's input ports

        Returns:
          A (dict) of module's input ports names to NeuralTypes mapping
        """
        return self._input_ports

    @property
    def output_ports(self) -> Optional[Dict[str, NeuralType]]:
        """Get all module's output ports

        Returns:
          A (dict) of module's output ports names to NeuralTypes mapping
        """
        return self._output_ports

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

    @staticmethod
    @abstractmethod
    def create_ports(**kwargs):
        """
        A static function that must be defined by the child Neural Module. It
        returns the input and output ports of the module.

        Returns:
            Two dictionaries, the first representing the input ports of the
            module and the second representing the output ports of the module
        """
        return {}, {}

    @staticmethod
    def update_local_params():
        """
        Loops through the call chain of class initializations and stops at the
        first class that is not an instance of Neural Module. At each step of
        the loop, the class contructor arguments are added to a dictionary
        containing the local parameters used to construct the Neural Module

        Returns:
            A dictionary containing all parameters passed to the module's init
            chain.
        """
        local_parameters = {}
        for frame in stack()[1:]:
            posname, kwname, localvars = getargvalues(frame[0])[-3:]
            # Check if caller is a Neural Module
            if ("self" in localvars and
                    isinstance(localvars["self"], NeuralModule)):
                if posname is not None:
                    raise ValueError("NeuralModules cannot accept `*` "
                                     "positional arguments.")
                # Get func arg dict
                localvars.update(localvars.pop(kwname, []))
                del localvars["self"]
                local_parameters.update(localvars)
            # Else we have rearched the end of the init callchain and we are
            # done
            else:
                break

        return local_parameters
