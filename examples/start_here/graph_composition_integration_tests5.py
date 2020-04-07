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

import torch

import inspect
import nemo
from nemo.core import NeuralGraph, OperationMode

logging = nemo.logging

nf = nemo.core.NeuralModuleFactory()
# Instantiate the necessary neural modules.
dl_training = nemo.tutorials.RealFunctionDataLayer(n=10000, batch_size=128)
fx = nemo.tutorials.TaylorNet(dim=4)
loss = nemo.tutorials.MSELoss()

logging.info(
    "This example shows how one can access modules nested in a graph."
)

# Build the training graph.
with NeuralGraph(operation_mode=OperationMode.both, name="trainable_module") as trainable_module:
    # Bind the input.
    _ = fx(x=trainable_module)
    # All outputs will be bound by default.

# Compose two graphs into final graph.
with NeuralGraph(operation_mode=OperationMode.training, name="training_graph") as training_graph:
    # Take outputs from the training DL.
    x, t = dl_training()
    # Pass them to the trainable module.
    p = trainable_module(x=x)
    # Pass both of them to loss.
    lss = loss(predictions=p, target=t)

print(trainable_module.list_modules())

print(training_graph.list_modules())

# Access modules.
dl_training_ref = training_graph["dl_training"]
fx_ref = training_graph["fx"]
loss_ref = training_graph["loss"]

# Throws an exception.
try:
    _ = training_graph["other_module"]
except KeyError as e:
    print("Got error: {}".format(e))
    pass