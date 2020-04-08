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

from nemo.utils.object_registry import ObjectRegistry

from nemo.core.neural_factory import OperationMode
from nemo.core.neural_graph import NeuralGraph


class NeuralGraphManager(ObjectRegistry):
    def __init__(self):
        """
            Constructor. Initializes the manager. Sets active graph to None.
        """
        super().__init__("graph")
        self._active_graph = None

    def summary(self):
        """ Prints a nice summary. """
        # TODO: a nicer summary. ;)
        desc = ""
        for graph in self:
            desc = desc + "`{}`: {}\n".format(graph.name, graph)
        return desc

    @property
    def active_graph(self):
        """
            Property returns the active graph. If there is no active graph, creates a new one.

            Returns:
                The active graph object.
        """
        # Create a new graph - training is the default.
        if self._active_graph is None:
            # Create a new "default" graph. Default mode: both.
            new_graph = NeuralGraph(operation_mode=OperationMode.both)
            new_graph.name = self.register(new_graph, None)
            # Set the newly created graph as active.
            self._active_graph = new_graph

        # Return the graph.
        return self._active_graph

    @active_graph.setter
    def active_graph(self, graph):
        """
            Property sets the active graph.

            Args:
                graph: Neural graph object that will become active.
        """
        # Activate the graph.
        self._active_graph = graph
