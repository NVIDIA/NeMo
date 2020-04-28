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

from nemo.core.neural_types import NeuralType
from nemo.utils import logging
from nemo.utils.module_port import ModulePort

# Actually this throws error as module dependency is: core depends on utils :]
# from nemo.core.neural_types import NeuralType


class GraphInput(object):
    """ A helper class represenging a single bound input. """

    def __init__(self, type):
        """ 
        Initializes object.

        Args:
            type: a NeuralType object.
        """
        # (Neural) Type of input.
        self._type = type
        # List of ModulePort tuples to which this input links to (module name, port name).
        self._consumers = []

    def bind(self, module_ports):
        """ Binds the (modules-ports) to this "graph input".

            Args:
                module_ports: A single ModulePort OR a list of ModulePort tuples to be added.
        """
        # Handle both single port and lists of ports to be bound.
        if type(module_ports) is not list:
            module_ports = [module_ports]
        # Interate through "consumers" on the list and add them to bound input.
        for module_port in module_ports:
            self._consumers.append(module_port)

    @property
    def type(self):
        """ Returns NeuralType of that input. """
        return self._type

    @property
    def consumers_ports(self):
        """ Returns list of bound modules (i.e. (module name, port name) tupes) """
        return self._consumers


class GraphInputs(MutableMapping):
    '''
        A specialized dictionary that contains bound inputs of a Neural Graph.
    '''

    def __init__(self):
        """ Initializes the mapping. """
        self._inputs = {}

    def __setitem__(self, key, value):
        """
            This method is used to "create" a bound input, i.e. copy definition from indicated module input port.

            Args:
                key: name of the input port of the Neural Graph.
                value: NeuralType that will be set.
        """
        if key in self._inputs.keys():
            raise KeyError("Overwriting definition of a previously bound port `{}` is not allowed".format(key))

        # Make sure that a proper NeuralType definition was passed here.
        if isinstance(value, NeuralType):
            val_type = value
        elif isinstance(value, GraphInput):
            val_type = value.type
        else:
            raise TypeError("Port `{}` definition must be must be a NeuralType or GraphInput type".format(key))
        # Ok, add definition to list of mapped (module, port)s.
        # Note: for now, there are no mapped modules, so copy only (neural) type.
        self._inputs[key] = GraphInput(type=val_type)

    def __getitem__(self, key):
        """ Returns bound input. """
        return self._inputs[key]

    def __delitem__(self, key):
        raise NotImplementedError("Deleting a bound input port is not allowed")

    def __iter__(self):
        """ Iterates over the bound inputs. """
        return iter(self._inputs)

    def __len__(self):
        """ Return number of bound inputs. """
        return len(self._inputs)

    @property
    def definitions(self):
        """ Property returns definitions of the input ports by extracting them on the fly from list. """
        # Extract port definitions (Neural Types) from the inputs list.
        return {k: v.type for k, v in self._inputs.items()}

    def has_binding(self, module_name, port_name):
        """ 
            Checks if there is a binding leading to a given module and its given port. 

            Returns:
                key in the list of the (bound) input ports that leads to a given module/port or None if the binding was
                not found.
        """
        for key, binding in self._inputs.items():
            for (module, port) in binding.consumers_ports:
                if module == module_name and port == port_name:
                    return key
        return None

    def serialize(self):
        """ Method responsible for serialization of the graph inputs.

            Returns:
                List containing mappings (input -> module.input_port).
        """
        serialized_inputs = []
        # Iterate through "bindings".
        for key, binding in self._inputs.items():
            for (module, port) in binding.consumers_ports:
                # Serialize: input -> module.port.
                target = module + "." + port
                serialized_inputs.append(key + "->" + target)
        # Return the result.
        return serialized_inputs

    @classmethod
    def deserialize(cls, serialized_inputs, modules, definitions_only=False):
        """ 
            Class method responsible for deserialization of graph inputs.

            Args:
                serialized_inputs: A list of serialized inputs in the form of ("input->module.input_port")
                modules: List of modules required for neural type copying/checking.
                definitions_only: deserializes/checks only the definitions, without binding the consumers.

            Returns:
                Dictionary with deserialized inputs.
        """
        inputs = GraphInputs()
        # Iterate through serialized inputs one by one.
        for i in serialized_inputs:
            # Deserialize!
            [key, consumer] = i.split("->")
            [consumer_name, consumer_port_name] = consumer.split(".")
            # Add the input.
            if key not in inputs.keys():
                # Get neural type from module input port definition.
                n_type = modules[consumer_name].input_ports[consumer_port_name]
                # Create a new input.
                inputs[key] = n_type
            # Optionally, also bind the "consumers".
            if not definitions_only:
                # Bound input.
                inputs[key].bind(ModulePort(consumer_name, consumer_port_name))
        # Done.
        return inputs
