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

# Sadly have to import the whole "nemo" python module to avoid circular dependencies.
# Moreover, at that point nemo module doesn't contain "core", so during "python module registration"
# nothing from nemo.core, including e.g. types (so we cannot use them for "python 3 type hints").
from nemo.utils.neural_graph.object_registry import ObjectRegistry


class NeuralGraphManager(ObjectRegistry):
    def __init__(self):
        """
            Constructor. Initializes the manager. Sets active graph to None.
        """
        super().__init__("graph")
        self._active_graph = None

    def __eq__(self, other):
        """
            Checks if two managers have the same content.
            Args:
                other: A second manager object.
        """
        if not isinstance(other, ObjectRegistry):
            return False
        return super().__eq__(other)

    def summary(self) -> str:
        """
            Returns:
                A summary of the graphs on the list.
        """
        # Line "decorator".
        summary = "\n" + 113 * '=' + "\n"
        summary += "Registry of {}s:\n".format(self._base_type_name)
        for graph in self:
            summary += " * {} ({}) [{}]\n".format(graph.name, len(graph), graph.operation_mode)
        # Line "decorator".
        summary += 113 * '='
        return summary

    @property
    def active_graph(self) -> "NeuralGraph":
        """
            Property returns the active graph. If there is no active graph, creates a new one.

            Returns:
                The active graph object.
        """
        # Create a new graph - training is the default.
        if self._active_graph is None:
            # Import core here (to avoid circular dependency between core-utils).
            from nemo.core import NeuralGraph, OperationMode

            # Create a new "default" graph. Default mode: both.
            new_graph = NeuralGraph(operation_mode=OperationMode.both)
            new_graph._name = self.register(new_graph, None)
            # Set the newly created graph as active.
            self._active_graph = new_graph

        # Return the graph.
        return self._active_graph

    @active_graph.setter
    def active_graph(self, graph: "NeuralGraph"):
        """
            Property sets the active graph.

            Args:
                graph: Neural graph object that will become active.
        """
        # Activate the graph.
        self._active_graph = graph
