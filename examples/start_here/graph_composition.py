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

# This will create a default graph: "training"
_, _ = dl()


with NeuralGraph(operation_mode=OperationMode.inference) as validation_dl:
    x_valid, y_valid = dl()

# Build the training graph.
with NeuralGraph(operation_mode=OperationMode.training, name="trainable_module") as trainable_module:
    # Add modules to graph.
    # Bind the first input.
    p1 = fx1(x=trainable_module)
    p2 = fx2(x=p1)
    p3 = fx2(x=p2)
    # All outputs will be binded by default.

# Compose two graphs into final graph.
with NeuralGraph(operation_mode=OperationMode.training, name="training_graph") as training_graph:
    # Take outputs from the first graph.
    x, y = AppState().graphs["training"]()
    # Pass them to the second graph.
    _, _, p = trainable_module(x)
    # Pass both of them to loss.
    lss = loss(predictions=p, target=y)


# Show all graphs.
print(AppState().graphs.summary())

# Show details of graph containing the trainable training module.
# print(training_module)

# Show
print(AppState().graphs["training1"])
# print(training_graph2)

# SimpleLossLoggerCallback will print loss values to console.
callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[lss], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}'),
)

# Invoke "train" action.
nf.train([lss], callbacks=[callback], optimization_params={"num_epochs": 3, "lr": 0.0003}, optimizer="sgd")
