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

from nemo.utils import logging
from nemo.utils.module_port import ModulePort


class GraphOutput(object):
    """ A helper class represenging a single bound output. """

    def __init__(self, type, producer_port):
        """ 
        Initializes object.

        Args:
            type: a NeuralType object.
            producer_port: a producer ModulePort tuple (module name, port name).
        """
        self._type = type
        self._producer_port = producer_port

    @property
    def type(self):
        """ Returns NeuralType of that output. """
        return self._type

    @property
    def producer_port(self):
        """ Returns producer port (module name, port name) tuple. """
        return self._producer_port


class GraphOutputs(MutableMapping):
    '''
        A specialized dictionary that contains bound outputs of a Neural Graph.
        In fact stores two lists of "outputs":
            - "default" outputs with default keys taken from outputs of modules (might result in
            overwriting some keys), and
            - "manual" used for specifying the subset of outputs, each with a new/different key
        When accessing the outputs, it returns the "manual" outputs. If "manual" outputs are not defined,
        will return/work on "default" outputs.
    '''

    def __init__(self, tensors_list):
        """ Initializes two (empty) dictionaries. """

        # List of tensors - passed from the external neural graph object.
        self._tensors_list = tensors_list

        # This dictionary stores the output tensors collected during the "default" tensor recording.
        # As they are using the default port names, the second/next tensor published on the same port
        # will overwrite the old one (Warning).
        self._default_outputs = {}

        # This dictionary stores list of output tensors of module "manually" indicated by the user.
        # In this case tring to overwriting the existing ports with new tensors will be forbidden (Exception).
        self._manual_outputs = {}

    def __setitem__(self, key, value):
        """
            This method is used to set the manual output - creates a GraphOutput item and adds it to the list.
            
            Args:
                key: name of the output (port).
                value: tensor that will be used to create GraphOutput.
        """
        # Make sure that user passed a NmTensor.
        assert type(value).__name__ == "NmTensor"
        if key in self._manual_outputs.keys():
            raise KeyError("Overwriting of a port `{}` that was previously manually bound is not allowed".format(key))
        # Ok, set output.
        self._manual_outputs[key] = GraphOutput(value.type, value.producer_port)

    def __getitem__(self, key):
        """ Returns GraphOutput - depending whether there are some manual outputs or not. """
        if len(self._manual_outputs) > 0:
            return self._manual_outputs[key]
        else:  # Use default dict.
            return self._default_outputs[key]

    def __delitem__(self, key):
        raise NotImplementedError("Deleting a bound output is not allowed")

    def __iter__(self):
        """ Iterates over the outputs - depending whether there are some manual outputs or not. """
        if len(self._manual_outputs) > 0:
            return iter(self._manual_outputs)
        else:  # Use default dict.
            return iter(self._default_outputs)

    def __len__(self):
        """ Return number of outputs - depending whether there are some manual outputs or not. """
        if len(self._manual_outputs) > 0:
            return len(self._manual_outputs)
        else:  # Use default dict.
            return len(self._default_outputs)

    def bind(self, tensors_list, port_names=None):
        """ Binds the default outputs.

            Args:
                tensors_list: List of tensors to be added.
                port_names: List of port names (visible outside). If None: using internal tensor "output port names".
        """
        # Set names.
        if port_names is None:
            port_names = [tensor.name for tensor in tensors_list]

        for name, tensor in zip(port_names, tensors_list):
            # Check the presence of the port name in "default" dictionary.
            if name in self._default_outputs.keys():
                # Name present - use the name being combination of producer and port names.
                name = tensor.producer_name + "_" + tensor.name

                logging.warning(
                    "Setting unigue name of the default output port `{}` produced by `{}` to `{}`".format(
                        tensor.name, self._default_outputs[tensor.name]._producer_port.module_name, name
                    )
                )
            # Still, "overwrite" it.
            self._default_outputs[name] = GraphOutput(tensor.type, tensor.producer_port)

    @property
    def definitions(self):
        """ Property returns definitions of the output ports by extracting them on the fly from the bound outputs. """
        # Get the right output dictionary.
        d = self._manual_outputs if len(self._manual_outputs) > 0 else self._default_outputs

        # Extract port definitions (Neural Types).
        return {k: v.type for k, v in d.items()}

    @property
    def tensors(self):
        """ Property returns output tensors by extracting them on the fly from the bound outputs. """
        # Get the right output dictionary.
        d = self._manual_outputs if len(self._manual_outputs) > 0 else self._default_outputs

        output_tensors = {}
        # Get tensors by acessing the producer-ports.
        for k, v in d.items():
            producer_name = v.producer_port.module_name
            producer_port_name = v.producer_port.port_name
            # Find the right output tensor.
            tensor = self._tensors_list[producer_name][producer_port_name]
            # Add it to the dictionary.
            output_tensors[k] = tensor

        return output_tensors

    def serialize(self):
        """ Method responsible for serialization of the graph outputs.

            Returns:
                List containing mappings (module.output_port -> output).
        """
        serialized_outputs = {"outputs": []}

        # Get the right output dictionary.
        if len(self._manual_outputs) > 0:
            serialized_outputs["type"] = "manual"
            d = self._manual_outputs
        else:
            serialized_outputs["type"] = "default"
            d = self._default_outputs

        # Iterate through "bindings".
        for key, binding in d.items():
            # Serialize: module.port -> output.
            source = binding.producer_port.module_name + "." + binding.producer_port.port_name
            serialized_outputs["outputs"].append(source + "->" + key)
        # Return the result.
        return serialized_outputs


    def deserialize(self, serialized_outputs, modules):
        """ 
            Method responsible for deserialization of graph outputs.

            Args:
                serialized_outputs: A list of serialized outputs in the form of ("module.output_port->key")
                modules: List of modules required for neural type copying/checking.
        """
        # Check type.
        if serialized_outputs["type"] == "default":
            # We do not need to deserialize.
            # self._default_outputs will be recorded automatically during graph execution.
            # TODO: check neural types.
            return

        # Iterate through serialized inputs one by one.
        for i in serialized_outputs["outputs"]:
            # Deserialize!
            [producer, key] = i.split("->")
            [producer_name, producer_port_name] = producer.split(".")

            # Get neural type from module output port definition.
            n_type = modules[producer_name].output_ports[producer_port_name]
            # Create a new input.
            go = GraphOutput(n_type, ModulePort(producer_name, producer_port_name))
            self._manual_outputs[key] = go
            # TODO: check neural types.

        # Done.


