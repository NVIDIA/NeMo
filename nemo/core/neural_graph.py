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

from collections import namedtuple
from typing import Dict, Optional

from nemo.core import OperationMode
from nemo.core.neural_interface import NeuralInterface
from nemo.core.neural_types import (
    NeuralPortNameMismatchError,
    NmTensor,
)
from nemo.utils.bound_inputs import BoundInput, BoundInputs
from nemo.utils.bound_outputs import BoundOutputs


class NeuralGraph(NeuralInterface):
    """
        Neural Graph class stores dynamically defined graphs of connected Neural Modules.
    """

    def __init__(self, operation_mode=OperationMode.both, name=None):
        """
            Constructor. Initializes graph variables.

            Args:
                operation_mode: Graph operation mode, that will be propagated along modules during graph creation.
                [training | eval | both] (DEFAULT: both)
                name: Name of the graph (optional)
        """
        # Initialize the inferface.
        super().__init__(name)

        # Register graph.
        self._name = self._app_state.register_graph(self, name)

        # Store name and operation mode.
        self._operation_mode = operation_mode

        # Input ports and tensors - empty for now.
        self._bound_input_ports = {}
        self._bound_input_tensors = {}
        # List of modules of bound inputs - so we will update their output tensors when the "bound"
        # input port will be connected.
        self._bound_input_modules = {}

        # Bound inputs.
        self._bound_inputs = BoundInputs()

        # Bound outputs.
        self._bound_outputs = BoundOutputs()

        # "Modules" - list of modules constituting "nodes" in a given graph.
        self._modules = {}
        # "Steps": order of the  execution of modules in a graph.
        self._steps = []

    def __call__(self, **kwargs):
        """
            This method "nests" one existing neural graph into another one.
            Also checks if all inputs were provided and properly connects them.

        """
        # print(" Neural Graph {} __call__".format(self._name))

        # Test operation modes of the nested graphs.
        outer_mode = self._app_state.active_graph.operation_mode
        inner_mode = self.operation_mode

        if inner_mode == OperationMode.inference and outer_mode == OperationMode.training:
            raise TypeError("Cannot nest 'inference' graph into 'training'")

        if inner_mode == OperationMode.training and outer_mode == OperationMode.inference:
            raise TypeError("Cannot nest 'training' graph into 'inference'")

        if inner_mode == OperationMode.training and outer_mode == OperationMode.both:
            raise TypeError("Cannot nest 'training' graph into 'both'")

        if inner_mode == OperationMode.inference and outer_mode == OperationMode.both:
            raise TypeError("Cannot nest 'inference' graph into 'both'")

        # Get input and output ports definitions.
        input_port_defs = self.input_ports

        # "Copy" all the operations from the previous graph. TODO better!
        for step in self._steps:
            self._app_state.active_graph.record_step(*step)
        # print(self._steps)

        ###### PROCESS INPUTS. ######
        # Iterate through all passed parameters.
        for port_name, port_content in kwargs.items():
            # make sure that passed arguments correspond to input port names
            if port_name not in input_port_defs.keys():
                raise NeuralPortNameMismatchError(port_name)

            # Analogically to NeuralModule, at that point the input can be one of three types:
            # * NeuralGraph -> bind port using the default name and type.
            # * BoundInput -> check definition, if ok bind port
            # * NmTensor -> check definition, add self as a "consumer" of a tensor (produced by other module).

            # Check what was actually passed.
            if type(port_content) is NeuralGraph:

                # Make sure that port_content is the currently active graph!
                if port_content is not self._app_state.active_graph:
                    raise ConnectionError("Ports can be bound only by passing the active graph object!")

                # This case: we are nesting one graph into another and must bind input port of one graph in another!
                # So generally we will "copy" the BoundInput object.

                # Create an alias so the logic will be more clear.
                active_graph = port_content

                # Copy the  port "definition" (i.e. is NeuralType) using the same port name.
                # This might throw an exception if port with that name was already bound!
                active_graph.input_ports[port_name] = input_port_defs[port_name].type

                # Bind the neural graph input port, i.e. remember that a given graph port should pass data
                # to all "bound modules" (when it finally will be connected).
                active_graph.input_ports[port_name].bind(input_port_defs[port_name].modules)

                # Please note that there are no "consumers" here - this is a "pure" binding.

            elif type(port_content) is BoundInput:

                # Compare input port definition with the received definition.
                input_port_defs[port_name].type.compare_and_raise_error(
                    self.__class__.__name__, port_name, port_content.type
                )

                # Bind the input port modules, i.e. remember that a given graph port should pass data
                # to all "bound modules" (when it finally will be connected).
                port_content.bind(self.modules)

                # Please note that there are no "consumers" here - this is a "pure" binding.

            elif type(port_content) is NmTensor:
                # Compare input port definition with the received definition.
                input_port_defs[port_name].type.compare_and_raise_error(
                    self.__class__.__name__, port_name, port_content
                )

                # Reaching that point means that we accepted input to a bound graph port.
                # Need to connect it - "copy" all modules connected to this "bound input" as consumers.
                for consumer in input_port_defs[port_name].modules:
                    # Add consumer.
                    port_content.add_consumer(consumer.module_name, consumer.port_name)

                    # The current graph parsing requires us to update all outputs of
                    # a module that "accepted" the input.
                    # In other words: for every "consumer" we need to update all tensors it produced.
                    # Update means changing the original producer_args for ALL TENSORS IN THE GRAPH produced by
                    # this module.
                    producer_name = consumer.module_name
                    if producer_name in self._bound_outputs.all.keys():
                        # Get all tensor producer by this module.
                        for output_tensor in self._bound_outputs.all[producer_name]:
                            # Set "input port value" to new content - which indicates tensor (and producer)
                            # that will be used during graph backward traverse.
                            output_tensor.producer_args[port_name] = port_content  # i.e. Tensor.

            else:
                raise TypeError(
                    "Input '{}' can be of one of three types: NeuralGraph, BoundInput or NmTensor".format(port_name)
                )

        ###### PRODUCE OUTPUTS. ######
        # This part is different from Neural Module.
        # Now the goal is NOT to create NEW "tensors", but to return the BOUND ones!
        # Still, those must be bound in the outer (active) graph.
        if len(self._bound_outputs) == 1:
            # Return the single tensor.
            results = next(iter(self._bound_outputs.values()))

            # "Copy" all the tensors from the nested graph. TODO COPY??
            # Bind the "default" output ports.
            self._app_state.active_graph.bind_outputs([results])


        else:
            # Create a named tuple type enabling to access outputs by attributes (e.g. out.x).
            output_class_name = f'{self.__class__.__name__}Output'
            result_type = namedtuple(typename=output_class_name, field_names=self._bound_outputs.keys())

            # Return the "default" bound output ports.
            results = result_type(*self._bound_outputs.values())

        # Return the results.
        return results

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        .. note::
            This method is NOT returning the dictionary with definitions (like Neural Modules),
            but the BoundInputs object.
            This was required to enable user to bound inputs with the dict's __setitem__ construct.

        Returns:
            A graph bound input ports.
        """
        return self._bound_inputs

    @property
    def output_ports(self):
        """
            Returns module output ports.

        .. note::
            This method is NOT returning the dictionary with definitions (like Neural Modules),
            but the BoundOutputs object.
            This was required to enable user to override the "default bound outputs"
            with the dict's __setitem__ construct.

        Returns:
            A graph bound output ports.
            
        """
        return self._bound_outputs

    @property
    def operation_mode(self):
        """ Returns operation mode. """
        return self._operation_mode

    def __enter__(self):
        """ 
            Activates this graph.
        
            Returns:
                The graph object.
        """
        self._app_state.active_graph = self
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
            Deactivates the current graph.
        """
        self._app_state.active_graph = None

    def activate(self):
        """ 
            Activates this graph.
        """
        self._app_state.active_graph = self

    def deactivate(self):
        """
            Deactivates the current graph.
        """
        self._app_state.active_graph = None

    def __str__(self):
        """ Prints a nice summary. """
        # TODO: a nice summary. ;)
        desc = "`{}` ({}):\n".format(self.name, len(self._steps))
        for op in self._steps:
            desc = desc + "  {}\n".format(type(op[0]).__name__)
        return desc

    def __getitem__(self, key):
        """ Returns module given its name (name of the variable).

            Args:
                key: Name of the variable.
        """
        if key not in self._modules.keys():
            raise KeyError("Neural Graph doesn't contain a module named {}".format(key))
        return self._modules[key]

    def __len__(self):
        return len(self._modules)

    def list_modules(self):
        desc = "{} ({}):\n".format(self.name, len(self))
        for key, value in self._modules.items():
            desc += " * `{}` ({})\n".format(key, value)
        return desc

    def record_step(self, module, inputs):
        """
            Records the operation (module plus passed inputs) on a list.
        """
        # Check if module with that name already exists.
        # if module.name in self._modules.keys():
        #    raise KeyError("Neural Graph already contains a module named {}".format(module.name))
        # Add module to list of modules.
        self._modules[module.name] = module

        # Add step.
        self._steps.append([module, inputs])

    # def bind_input(self, port_name, port_definition, bound_module):
    #    # print("Binding input: `{}`: def = `{}` value = NONE".format(port_name, port_definition))
    #    # Copy the definition of the port to graph input port definition.
    #    self._bound_input_ports[port_name] = port_definition

    #    # Indicate that this tensor is missing and has to be provided!
    #    self._bound_input_tensors[port_name] = None
    #    # Additionally, remember the bound module
    #    self._bound_input_modules[port_name] = bound_module

    def bind_outputs(self, tensors_list):
        """ Binds the output tensors.

            Args:
                tensors_list: List of tensors to be bound.
        """
        self._bound_outputs.bind(tensors_list)

    def show_bound_inputs(self):
        print("bound input ports: ")
        for key, value in self._bound_input_ports.items():
            print(" * `{}`: `{}` ({})".format(key, value, type(value)))

        print("bound input tensors: ")
        for key, value in self._bound_input_tensors.items():
            print(" * `{}`: `{}` ({})".format(key, value, type(value)))

    def show_bound_outputs(self):
        print("bound (default) output ports: ")
        for key, value in self._bound_output_ports_default.items():
            print(" * `{}`: `{}` ({})".format(key, value, type(value)))

        print("bound (default) output tensors: ")
        for key, value in self._bound_output_tensors_default.items():
            print(" * `{}`: `{}` ({})".format(key, value, type(value)))
