# ! /usr/bin/python
# -*- coding: utf-8 -*-

# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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

from typing import Dict, Optional

from nemo.core.app_state import AppState
from nemo.core.neural_factory import NeuralType, OperationMode


class NeuralGraph:
    def __init__(self, operation_mode):
        """
            Constructor. Initializes graph variables.

            Args:
                operation_mode: Graph operation mode, that will be propagated along modules during graph creation.
                [training | eval]
        """
        print('__init__ called')
        self._operation_mode = operation_mode
        # Input and output ports - empty for now.
        self._binded_input_ports = {}
        self._binded_output_ports = {}

    @property
    def input_ports(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module input ports

        Returns:
          A (dict) of module's input ports names to NeuralTypes mapping
        """
        return self._binded_input_ports

    @property
    def output_ports(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports

        Returns:
          A (dict) of module's output ports names to NeuralTypes mapping
        """
        return self._binded_output_ports

    def __enter__(self):
        """ Activates given graph as current. """
        print('__enter__ called')
        AppState().active_graph = self
        # self._app_state.active_graph = self
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ Deactivates current graph. """
        print('__exit__ called')
        AppState().active_graph = None
        if exc_type:
            print(f'exc_type: {exc_type}')
            print(f'exc_value: {exc_value}')
            print(f'exc_traceback: {exc_traceback}')

    def record_step(self):
        pass
