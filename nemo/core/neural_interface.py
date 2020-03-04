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

import collections
from abc import ABC, abstractmethod
from typing import Dict, Optional

import nemo
from nemo.core.neural_types import (
    CanNotInferResultNeuralType,
    NeuralPortNameMismatchError,
    NeuralPortNmTensorMismatchError,
    NeuralType,
    NeuralTypeComparisonResult,
    NmTensor,
)


class NeuralInterface(ABC):
    """
        Abstract class offering interface shared between Neural Module and Neural Graph.
        Had to move it to a separate class to:
        a) avoid circular imports between Neural Module and Graph.
        b) avoid collection of init_params implemented by default in Neural Module.
        c) extract only the methods that are shared (NMs have plenty of methods that are not making any sense for 
        graph, e.g. get_weights, tie_weights, )
    """

    def __init__(self):
        self._app_state = nemo.core.app_state.AppState()
        pass

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
        # Get input and output ports definitions.
        input_port_defs = self.input_ports
        output_port_defs = self.output_ports

        # Handle a special case - one passes an instance of neural graph.
        # This should bind all ports.
        # if len(kwargs) == 1:  # and isinstance(kwargs.items()[1], NeuralGraph):
        # TODO: Those ports should be binded.
        # print("Binding all ports")
        # Not doing it now - assuming that user will manually indicate ports to bind.

        first_input_nmtensor_type = None
        input_nmtensors_are_of_same_type = True
        # Iterate through all passed parameters.
        for port_name, port_source in kwargs.items():
            # make sure that passed arguments correspond to input port names
            if port_name not in input_port_defs.keys():
                raise NeuralPortNameMismatchError("Wrong input port name: {0}".format(port_name))

            # Check what was actually passed.
            if isinstance(port_source, nemo.core.NeuralGraph):
                # Bind this input port to a neural graph.

                # TODO: make sure that port_source ==  self._app_state.active_graph ?????
                port_source.bind_input_ports(port_name, input_port_defs[port_name], self)
                # It is "compatible by definition";), so we don't have to check this port further.
            else:  # : port_source is a neural module.
                # Compare input port definition with the received definition.
                input_port = input_port_defs[port_name]
                type_comatibility = input_port.compare(port_source)
                if (
                    type_comatibility != NeuralTypeComparisonResult.SAME
                    and type_comatibility != NeuralTypeComparisonResult.GREATER
                ):
                    raise NeuralPortNmTensorMismatchError(
                        "\n\nIn {0}. \n"
                        "Port: {1} and a NmTensor it was fed are \n"
                        "of incompatible neural types:\n\n{2} \n\n and \n\n{3}"
                        "\n\nType comparison result: {4}".format(
                            self.__class__.__name__,
                            port_name,
                            input_port_defs[port_name],
                            port_source,
                            type_comatibility,
                        )
                    )
        # TODO CHECK 1: Are we making sure that ALL necessary inputs that were PASSED?

        # At that point we made sure that all ports are correcty connected.
        self._app_state.active_graph.record_operation(self, kwargs.items())

        # Here we will store the results.
        results = None

        if len(output_port_defs) == 1:
            out_name = list(output_port_defs)[0]
            out_type = output_port_defs[out_name]
            if out_type is None:
                if input_nmtensors_are_of_same_type:
                    out_type = first_input_nmtensor_type
                else:
                    raise CanNotInferResultNeuralType(
                        "Can't infer output neural type. Likely your inputs are of different type."
                    )
            # TODO CHECK 2: Why are we returning "something" (having input type) if there SHOULD be no output?
            results = NmTensor(producer=self, producer_args=kwargs, name=out_name, ntype=out_type,)
        else:
            result = []
            for out_port, n_type in output_port_defs.items():
                out_type = n_type
                if out_type is None:
                    if input_nmtensors_are_of_same_type:
                        out_type = first_input_nmtensor_type
                    else:
                        raise CanNotInferResultNeuralType(
                            "Can't infer output neural type. Likely your inputs are of different type."
                        )
                result.append(NmTensor(producer=self, producer_args=kwargs, name=out_port, ntype=out_type,))

            # Creating ad-hoc class for returning from module's forward pass.
            output_class_name = f'{self.__class__.__name__}Output'
            field_names = list(output_port_defs)
            result_type = collections.namedtuple(typename=output_class_name, field_names=field_names,)

            # Tie tuple of output tensors with corresponding names.
            results = result_type(*result)

        # Bind the output ports.
        self._app_state.active_graph.bind_output_ports(output_port_defs, results)

        # Return the results.
        return results
