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

logging = nemo.logging

nf = nemo.core.NeuralModuleFactory()
# Instantiate the necessary neural modules.
dl = nemo.tutorials.RealFunctionDataLayer(n=10000, batch_size=128)
m2 = nemo.tutorials.TaylorNet(dim=4)
loss = nemo.tutorials.MSELoss()


with NeuralGraph(operation_mode=OperationMode.training, name="g2") as g2:
    # Add module to graph and bind it input port 'x'.
    y = m2(x=g2)

print(g2)
g2.show_binded_inputs()
g2.show_binded_outputs()

# Build the training graph.
with NeuralGraph(operation_mode=OperationMode.training, name="g3") as g3:
    # Add modules to graph.
    x, t = dl()
    # Incorporate modules from existing graph
    p = g2(x=x)
    lss = loss(predictions=p, target=t)

print(g3)
g3.show_binded_inputs()
g3.show_binded_outputs()

# Show all graphs.
print(AppState().graphs.summary())

# SimpleLossLoggerCallback will print loss values to console.
callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[lss], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}'),
)

# Invoke "train" action.
nf.train([lss], callbacks=[callback], optimization_params={"num_epochs": 3, "lr": 0.0003}, optimizer="sgd")
