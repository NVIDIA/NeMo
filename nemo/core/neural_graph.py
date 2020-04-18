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

from collections import OrderedDict, namedtuple
from typing import Dict, Optional

from nemo.core import OperationMode
from nemo.core.neural_interface import NeuralInterface
from nemo.core.neural_types import NeuralPortNameMismatchError, NmTensor
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

        # "Modules" - list of modules constituting "nodes" in a given graph.
        self._modules = {}

        # All tensors produced within this graph (dict of dicts).
        # This stores  "all output tensors" dictionary, where the key is the name of "producer" module,
        # and the value contains a dictionary of all tensors produced by it.
        self._all_tensors = {}

        # "Steps": order of the  execution of modules in a graph.
        self._steps = OrderedDict()

        # Bound inputs.
        self._bound_inputs = BoundInputs()

        # Bound outputs.
        self._bound_outputs = BoundOutputs(self._all_tensors)

        # Flag indicating whether the "default" output ports/tensors will be automatically bound.
        self.default_output_binding = True

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

        # Check inputs: iterate through all inputs passed to the "self".
        for port_name, port_content in kwargs.items():
            # Make sure that passed arguments correspond to input port names.
            if port_name not in self.input_ports.keys():
                raise NeuralPortNameMismatchError(port_name)

        # "Nest" this graph into an active graph.
        results = self._app_state.active_graph.nest(self, kwargs)

        # Return output tensors.
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
    def modules(self):
        """ Returns modules. """
        return self._modules

    @property
    def steps(self):
        """ Returns steps. """
        return self._steps

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

    def record_step(self, module):
        """
            Records the operation (module plus passed inputs) on a list.
        """
        # Check if module with that name already exists.
        # TODO: Uncomment when we will refactor all examples so training/validation graphs won't be added
        # to the "default" graph.
        # if module.name in self._modules.keys():
        #    raise KeyError("Neural Graph already contains a module named {}".format(module.name))
        # Add module to list of modules.
        self._modules[module.name] = module

        # Add step - store the module name.
        self._steps[len(self._steps)] = module.name

    def unused_input_processing(self):
        """
            Old implementation! To be deleted!!!! Wrrrrrr!!!
        """
        # Check inputs: iterate through all inputs passed to the inner graph.
        for port_name, port_content in inner_graph_args.items():
            # Make sure that passed arguments correspond to input port names.
            if port_name not in inner_graph.input_ports.keys():
                raise NeuralPortNameMismatchError(port_name)

            # Analogically to NeuralModule, at that point the input can be one of three types:
            # * NeuralGraph -> bind port using the default name and type.
            # * BoundInput -> check definition, if ok bind port.
            # * NmTensor -> check definition, add self as a "consumer" of a tensor (produced by other module).

            # Check what was actually passed.
            if type(port_content) is NeuralGraph:

                # Make sure that port_content is the currently active graph, i.e. THIS GRAPH!
                if port_content is not self:
                    raise ConnectionError("Ports can be bound only by passing the active graph object!")

                # This case: we are nesting one graph into another and must bind input port of one graph in another!
                # So generally we will "copy" the BoundInput object, using the same name.

                # Copy the port "definition" (i.e. is NeuralType) using the same port name.
                # This might throw an exception if port with that name was already bound!
                self.input_ports[port_name] = inner_graph.input_ports[port_name].type

                # Remember that a given graph port should pass data to all "bound modules" of the "nested" graph
                # (when it finally will be connected).
                self.input_ports[port_name].bind(inner_graph.input_ports[port_name].modules)

                # Please note that there are no "consumers" here - this is a "pure" binding.

            elif type(port_content) is BoundInput:
                # Check if BoundInput belongs to the outer graph (i.e. self)!
                own_port = False
                for port in self.input_ports.items():
                    if port is BoundInput:
                        own_port = True
                        break
                if not own_port:
                    raise NeuralPortNameMismatchError(port_name)

                # Compare input port definition with the received definition.
                port_content.type.compare_and_raise_error(
                    self.__class__.__name__, port_name, inner_graph.input_ports[port_name].type
                )

                # Remember that a given graph port should pass data to all "bound modules" of the "nested" graph
                # (when it finally will be connected).
                port_content[port_name].bind(inner_graph.input_ports[port_name].modules)

                # Please note that there are no "consumers" here - this is a "pure" binding.

            elif type(port_content) is NmTensor:

                # Compare input port definition onf the inner graph with the received definition.
                inner_graph.input_ports[port_name].type.compare_and_raise_error(
                    self.__class__.__name__, port_name, port_content.type
                )
                # Note that this tensor is already! a part of this graph.
                # (It has to be output of one of the previous modules.)
                # So we can find it in: self._tensors

                # Reaching that point means that we accepted input to a bound graph port.
                # Need to connect it - "copy" all modules connected to this "bound input" as consumers.
                for consumer in inner_graph.input_ports[port_name].modules:
                    # Add consumer.
                    port_content.add_consumer(consumer.module_name, consumer.port_name)

                    # The current graph parsing requires us to update all outputs of
                    # a module that "accepted" the input.
                    # In other words: for every "consumer" we need to update all tensors it produced.
                    # Update means changing the original producer_args for ALL TENSORS IN THE GRAPH produced by
                    # this module.
                    # producer_name = consumer.module_name
                    # if producer_name in self._tensors.keys():
                    #    # Get all tensor producer by this module.
                    #    for output_tensor in self._tensors[producer_name]:
                    #        # Set "input port value" to new content - which indicates tensor (and producer)
                    #        # that will be used during graph backward traverse.
                    #        output_tensor.producer_args[port_name] = port_content  # i.e. Tensor.

            else:
                raise TypeError(
                    "Input '{}' can be of one of three types: NeuralGraph, BoundInput or NmTensor".format(port_name)
                )

    def nest(self, inner_graph, inner_graph_args):
        """
            Method nests (copies) a graph: modules, steps, topology (tensors).

            Args:
                inner_graph: Graph to be copied (will be "nested" in this (self) graph).
                inner_graph_args: inputs passed to the graph call.
        """
        # "Copy" modules.
        for key, module in inner_graph.modules.items():
            # Check if module with that name already exists.
            # TODO: Uncomment when we will refactor all examples so training/validation graphs won't be added
            # to the "default" graph.
            # if key in self._modules.keys():
            #    raise KeyError("Neural Graph already contains a module named {}".format(module.name))
            self._modules[key] = module

        # Next we should copy the topography - i.e. produce "real" copies of tensors.
        # In fact, instead of copying, we will produce them, following:
        # - the execution order defined in "steps"
        # - connectivity defined in tensor' consumers-ports
        # (so the same logic that will be used in graph deserialization)

        # So let us first serialize the connections of the nested graph.
        # Create a list: (producer.port -> consumer.port)
        inner_connections = []
        for tensors in inner_graph.tensors.values():
            for t in tensors.values():
                inner_connections.extend(t.connections())

        # We need to disable the binding of "defeault" ports on per module basis - we will "manually" produce
        # them only for ports that are already indicated as the "bound" ones in the inner graph.
        self.default_output_binding = False

        # Now "copy" steps, i.e. append them to the list, following the original order.
        # Moreover, actually execute each step.
        for _, module_name in inner_graph.steps.items():
            # Add step - store the module name.
            self._steps[len(self._steps)] = module_name

            # Get the module.
            module = self._modules[module_name]

            # Produce list of arguments that will be passed to a given modules.
            module_args = {}
            # Do it by:
            # - harvesing input port names of a given module,
            # - checking if the input was not bound (in the inner graph),
            # - checking if we have already tensors leading to that input (in outer graph).
            for input_port_name in module.input_ports.keys():
                # Check if this port was bound in the inner graph.
                key = inner_graph.input_ports.has_binding(module_name, input_port_name)
                # If so, then we must pass whatever was passed to that port in the list of arguments.
                if key is not None:
                    module_args[input_port_name] = inner_graph_args[key]
                    # As a result, the "module" call() will bind this input!
                    continue

                # Else: find a tensor that should be passed to the given module's input.
                # Search for producer/port that we should use.
                for connection in inner_connections:
                    if (
                        connection.consumer.module_name == module_name
                        and connection.consumer.port_name == input_port_name
                    ):
                        # Got the connection!
                        producer_name = connection.producer.module_name
                        producer_port_name = connection.producer.port_name
                        break
                # Now, the tensor is already produced in outer (i.e. this) graph!
                module_args[input_port_name] = self.tensors[producer_name][producer_port_name]

            #import pdb;pdb.set_trace()
            # Ok, now we have all keyword arguments. We can call() the module.
            # This will collect all the produced output tensors and add them to this graph.
            module(**module_args)

        # At that point we have all modules, steps and tensors added to outer (self) graph.
        # Now we have to prepare the outputs.

        # This part is different from Neural Module.
        # Now the goal is NOT to create NEW "tensors", but to return the BOUND ones!
        # Still, those must be bound in the outer (active) graph.

        # Get list of "the adequate output tensors".
        output_tensors = {}
        # Iterate through outputs of the inner graph.
        for key, tensor in inner_graph._bound_outputs.tensors.items():
            # Find the tensors within this (outer) graph that are outpus by the same producer-port.
            producer_name = tensor.producer_name
            producer_port_name = tensor.name
            # Get adequate tensor from "outer graph" (self).
            output_tensors[key] = self.tensors[producer_name][producer_port_name]

        if len(output_tensors) == 1:
            # Return a single tensor.
            results = next(iter(output_tensors.values()))

            # Bind the "default" output ports of the inner graph as "default" output ports of this graph.
            # Call the bind() method of bound_outputs directly, as we already have the tensors in our graph.
            self._bound_outputs.bind([results])

        else:
            # Create a named tuple type enabling to access outputs by attributes (e.g. out.x).
            output_class_name = f'{self.__class__.__name__}Output'
            result_type = namedtuple(typename=output_class_name, field_names=output_tensors.keys())

            # Return the bound output tensors.
            results = result_type(*output_tensors.values())

            # Bind the "default" output ports of the inner graph as "default" output ports of this graph.
            # Call the bind() method of bound_outputs directly, as we already have the tensors in our graph.
            self._bound_outputs.bind(output_tensors.values())

        # Ok, now we can turn automatic binding on.
        self.default_output_binding = True

        # Return the results.
        return results

    def bind_outputs(self, tensors_list):
        """ Binds the output tensors.

            Args:
                tensors_list: List of tensors to be bound.
        """
        # Add tensors list to of tensors.
        for tensor in tensors_list:
            # Add tensor to "all" tensors dictionary.
            producer_name = tensor.producer_name
            if producer_name not in self._all_tensors.keys():
                self._all_tensors[producer_name] = {}

            port_name = tensor.name
            # Add tensor.
            self._all_tensors[producer_name][port_name] = tensor

        # Bind the tensors as graph outputs.
        if self.default_output_binding:
            self._bound_outputs.bind(tensors_list)

    @property
    def tensors(self):
        """ Returns the dictionary of all output tensors, aggregated by modules (key). """
        return self._all_tensors

    def show_bound_inputs(self):
        print("bound input ports: ")
        # for key, value in self._bound_input_ports.items():
        #    print(" * `{}`: `{}` ({})".format(key, value, type(value)))

        print("bound input tensors: ")
        # for key, value in self._bound_input_tensors.items():
        #    print(" * `{}`: `{}` ({})".format(key, value, type(value)))

    def show_bound_outputs(self):
        print("bound (default) output ports: ")
        # for key, value in self._bound_output_ports_default.items():
        #    print(" * `{}`: `{}` ({})".format(key, value, type(value)))

        print("bound (default) output tensors: ")
        # for key, value in self._bound_output_tensors_default.items():
        #    print(" * `{}`: `{}` ({})".format(key, value, type(value)))
