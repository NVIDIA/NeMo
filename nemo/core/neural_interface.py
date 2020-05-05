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

from abc import ABC, abstractmethod
from typing import Dict

import nemo
from nemo.core.neural_types import NeuralType


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
        """
            Constructor. Creates a "shortcut" to the application state.
        """
        # Create access to the app state.
        self._app_state = nemo.utils.app_state.AppState()

    @property
    @abstractmethod
    def input_ports(self) -> Dict[str, NeuralType]:
        """ Returns definitions of module input ports

        Returns:
          A (dict) of module's input ports names to NeuralTypes mapping
        """

    @property
    @abstractmethod
    def output_ports(self) -> Dict[str, NeuralType]:
        """ Returns definitions of module output ports

        Returns:
          A (dict) of module's output ports names to NeuralTypes mapping
        """

    @abstractmethod
    def __call__(self, **kwargs):
        """
            This method is used during the construction of a graph for neural type compatibility checking.
            Actual implementation lies in Neural Module and Neural Graph classes.

        Returns:
          NmTensor object or tuple of NmTensor objects
        """

    @property
    def name(self):
        """ Returns the object name. """
        return self._name
