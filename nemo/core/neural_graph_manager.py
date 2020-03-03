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

# from collections.abc import Mapping

from nemo.core.neural_factory import OperationMode
from nemo.core.neural_graph import NeuralGraph


class NeuralGraphManager(object):
    def __init__(self):
        """
            Constructor. Initializes the manager.

            Args:
                operation_mode: Graph operation mode, that will be propagated along modules during graph creation.
                [training | eval]
        """
        self._active_graph = None
        self._graphs = {}

    def register_graph(self, graph):
        """ Registers a new graph. """
        # Create a unigue name.
        print("Name: ", graph._name)
        # Add it to the list.
        unique_name = self.__generate_unique_graph_name(graph._name)
        print("Unique name: ", unique_name)
        self._graphs[unique_name] = graph

    @property
    def graphs(self):
        """ Property returns the list of graphs.

            Returns:
                List of created graphs.
        """
        return self._graphs

    def summary(self):
        """ Prints a nice summary. """
        # TODO: a nice summry. ;)
        desc = ""
        for name, graph in self._graphs.items():
            desc = desc + "`{}`: {}\n".format(name, graph)
        return desc

    @property
    def active_graph(self):
        """
            Property returns the active graph. If there is no active graph, creates a new one.

            Returns:
                Active graph
        """
        # Create a new graph - training is the default.
        if self._active_graph is None:
            # Store graph on the list.
            new_graph = NeuralGraph(operation_mode=OperationMode.training)
            self.register_graph(new_graph)
            # Set the newly created graph as active.
            self._active_graph = new_graph

        # Return the graph.
        return self._active_graph

    @active_graph.setter
    def active_graph(self, graph):
        """ Property sets the active graph.

            Args:
                graph: Neural graph object that will become active.
        """
        # Activate the graph.
        self._active_graph = graph

    def __generate_unique_graph_name(self, name):
        """ Generates a new unique name by adding postfix (number). """
        # Simply return the same name as long as it is unique.
        if name not in self._graphs.keys():
            return name

        # Iterate through numbers.
        postfix = 1
        new_name = name + str(postfix)
        while new_name in self._graphs.keys():
            postfix = postfix + 1
            new_name = name + str(postfix)
        return new_name

    def __len__(self):
        """

        :return: Number of created neural graphs.

        """
        return len(self._graphs)

    def __getitem__(self, key):
        """
        Value getter function.

        :param key: Graph name.

        :return: Associated graph.
        """
        # Retrieve the value.
        return self._graphs[key]
