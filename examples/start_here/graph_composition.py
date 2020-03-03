# ! /usr/bin/python
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

import nemo
from nemo.core import AppState, NeuralGraph, NeuralModule, NeuralModuleFactory, OperationMode

nf = nemo.core.NeuralModuleFactory()
# Instantiate the necessary neural modules.
dl = nemo.tutorials.RealFunctionDataLayer(n=10000, batch_size=128)
fx1 = nemo.tutorials.TaylorNet(dim=4)
fx2 = nemo.tutorials.TaylorNet(dim=4)
loss = nemo.tutorials.MSELoss()

# This will create a default graph.
x_valid, y_valid = dl()


with NeuralGraph(operation_mode=OperationMode.training) as training_graph1:
    x, y = dl()

# Build the training graph.
with NeuralGraph(operation_mode=OperationMode.training) as training_graph2:
    # Add modules to graph.
    p1 = fx1(x=x)
    p2 = fx2(x=p1)
    p3 = fx2(x=p2)
    lss = loss(predictions=p3, target=y)


print(AppState().graphs.summary())

# print(training_graph1)
# print(training_graph2)

print(AppState().graphs["training1"])
