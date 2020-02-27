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

import nemo
from nemo.core import AppState, NeuralModule, NeuralModuleFactory, OperationMode

logging = nemo.logging


class NeuralGraph:
    def __init__(self, operation_mode):
        """
            Constructor. Initializes graph variables.

            Args:
                operation_mode: Graph operation mode, that will be propagated along modules during graph creation.
                [training | eval]
        """
        print('__init__ called')
        self.operation_mode = operation_mode
        self.app_state = AppState()

    def __enter__(self):
        print('__enter__ called')
        # Record itself as the current graph.
        self.app_state.active_graph = self
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        print('__exit__ called')
        # Deactivate current graph.
        self.app_state.active_graph = None
        if exc_type:
            print(f'exc_type: {exc_type}')
            print(f'exc_value: {exc_value}')
            print(f'exc_traceback: {exc_traceback}')

    def add_two(self):
        self.init_var += 2


nf = nemo.core.NeuralModuleFactory()
# Instantiate the necessary neural modules.
dl = nemo.tutorials.RealFunctionDataLayer(n=10000, batch_size=128)
fx = nemo.tutorials.TaylorNet(dim=4)
loss = nemo.tutorials.MSELoss()

# Build the training graph.
with NeuralGraph(operation_mode=OperationMode.training) as training_graph:
    print('inside with statement body')
    # Describe the activation flow.
    x, y = dl()
    p = fx(x=x)
    lss = loss(predictions=p, target=y)
