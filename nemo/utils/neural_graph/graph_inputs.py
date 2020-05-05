# -*- coding: utf-8 -*-

# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

from collections.abc import MutableMapping
from typing import Dict, List, Optional, Union

from frozendict import frozendict

from nemo.utils import logging
from nemo.utils.neural_graph.connection import StepModulePort


class GraphInput(object):
    """ A helper class represenging a single bound input. """

    def __init__(self, ntype: "NeuralType"):
        """ 
        Initializes object.

        Args:
            ntype: a NeuralType object.
        """
        # (Neural) Type of input.
        self._ntype = ntype
        # List of StepModulePort tuples to which this input links to (step number, module name, port name).
        self._consumers = []

    def bind(self, step_module_ports: Union[StepModulePort, List[StepModulePort]]):
        """ Binds the (step-module-ports) to this "graph input".
            Add "consumers" of this graph input (modules attached to this port),
            so when one actually will pass the NmTensor, those modules will be connected.

            Args:
                step_module_ports: A single StepModulePort OR a list of StepModulePort tuples to be added.
        """
        # Handle both single port and lists of ports to be bound.
        if type(step_module_ports) is not list:
            step_module_ports = [step_module_ports]
        # Interate through "consumers" on the list and add them to bound input.
        for smp in step_module_ports:
            self._consumers.append(smp)

    @property
    def ntype(self) -> "NeuralType":
        """
            Returns:
                NeuralType of a given input.
        """
        return self._ntype

    @property
    def consumers(self) -> List[StepModulePort]:
        """ 
            Returns:
                List of bound modules i.e. (step number, module name, port name) tupes.
        """
        return self._consumers


class GraphInputs(MutableMapping):
    '''
        A specialized dictionary that contains bound inputs of a Neural Graph.
    '''

    def __init__(self):
        """
            Initializes an empty dictionary.
        """
        self._inputs = {}

    def __setitem__(self, key: str, value: Union["NeuralType", GraphInput]):
        """
            This method is used to "create" a bound input, i.e. copy definition from indicated module input port.

            Args:
                key: name of the input port of the Neural Graph.
                value: NeuralType (or GraphInput) that will be set.
            
            Raises:
                KeyError: Definition of a previously bound port is not allowed.
                TypeError: Port definition must be must be a NeuralType or GraphInput type.
        """
        # Make sure that a proper object was passed here.
        if type(value).__name__ == "NeuralType":
            ntype = value
        elif isinstance(value, GraphInput):
            ntype = value.ntype
        else:
            raise TypeError("Port `{}` definition must be must be a NeuralType or GraphInput type".format(key))

        if key in self._inputs.keys():
            if self._inputs[key].ntype == ntype:
                raise KeyError("Overwriting definition of a previously bound port `{}` is not allowed".format(key))
            # Else: do nothing.
        else:
            # Ok, add definition to list of mapped (module, port)s.
            # Note: for now, there are no mapped modules, so copy only the (neural) type.
            self._inputs[key] = GraphInput(ntype=ntype)

    def __getitem__(self, key: str) -> GraphInput:
        """
            Returns the bound input associated with the given key.

            Args:
                key: Name of the bound input.
        """
        return self._inputs[key]

    def __delitem__(self, key: str):
        """
            Raises:
                TypeError as deletion of a bound input port is not allowed.
        """
        raise TypeError("Deletion of a bound input port is not allowed")

    def __iter__(self):
        """ 
            Returns:
                Iterator over the dict of bound inputs.
        """
        return iter(self._inputs)

    def __len__(self) -> int:
        """
            Return:
                The number of bound inputs.
        """
        return len(self._inputs)

    @property
    def definitions(self) -> Dict[str, "NeuralType"]:
        """
            Property returns definitions of the input ports by extracting them on the fly from list.

            ..info:
                This property actually returns a FrozenDict containing port definitions to indicate that
                port definitions SHOULD not be used during the actual binding.
            
            Returns:
                Dictionary of neural types associated with bound inputs.
        """
        # Extract port definitions (Neural Types) from the inputs list.
        return frozendict({k: v.ntype for k, v in self._inputs.items()})

    def has_binding(self, step_number: int, port_name: str) -> Optional[str]:
        """ 
            Checks if there is a binding leading to a given step number (module) and its given port. 
            (module name is redundant, thus skipped in this test).

            Returns:
                key in the list of the (bound) input ports that leads to a given step (module)/port
                or None if the binding was not found.
        """
        for key, binding in self._inputs.items():
            for (step, _, port) in binding.consumers:
                if step == step_number and port == port_name:
                    return key
        # Binding not found.
        return None

    def serialize(self) -> List[str]:
        """ Method responsible for serialization of the graph inputs.

            Returns:
                List containing mappings (input -> step.module.input_port).
        """
        serialized_inputs = []
        # Iterate through "bindings" (GraphInputs).
        for key, binding in self._inputs.items():
            # Get type.
            ntype_str = str(binding.ntype)
            for (step, module, port) in binding.consumers:
                # Serialize: input -> step.module.port | ntype
                target = str(step) + "." + module + "." + port
                # Serialize!
                serialized_inputs.append(key + "->" + target + " | " + ntype_str)
        # Return the result.
        return serialized_inputs

    @classmethod
    def deserialize(cls, serialized_inputs: List[str], modules: Dict[str, 'NeuralModule']):
        """ 
            Class method responsible for deserialization of graph inputs.

            Args:
                serialized_inputs: A list of serialized inputs in the form of ("input->module.input_port")
                modules: List of modules required for neural type copying/checking.

            Returns:
                Dictionary with deserialized inputs.
        """
        inputs = GraphInputs()
        # Iterate through serialized inputs one by one.
        for i in serialized_inputs:
            # Deserialize!
            [key, consumer_ntype] = i.split("->")
            [consumer, ntype_str] = consumer_ntype.split(" | ")
            [consumer_step, consumer_name, consumer_port_name] = consumer.split(".")
            # Add the input.
            if key not in inputs.keys():
                # Get neural type from module input port definition.
                ntype = modules[consumer_name].input_ports[consumer_port_name]
                # Make sure the graph bound  port type matches the deserialized type.
                assert ntype_str == str(ntype)

                # Create a new input.
                inputs[key] = ntype
            # Bind the "consumers".
            inputs[key].bind(StepModulePort(int(consumer_step), consumer_name, consumer_port_name))
        # Done.
        return inputs
