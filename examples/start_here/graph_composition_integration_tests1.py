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


with NeuralGraph(operation_mode=OperationMode.training, name="g1") as g1:
    x, t = dl()
    y = m2(x=x)

print(g1)
g1.show_binded_inputs()
g1.show_binded_outputs()

with NeuralGraph(operation_mode=OperationMode.training, name="g1.1") as g11:
    x1, t1, p1 = g1()
    lss = loss(predictions=p1, target=t1)


print(g11)
g11.show_binded_inputs()
g11.show_binded_outputs()


# Show all graphs.
print(AppState().graphs.summary())

# SimpleLossLoggerCallback will print loss values to console.
callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[lss], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}'),
)

# Invoke "train" action.
nf.train([lss], callbacks=[callback], optimization_params={"num_epochs": 3, "lr": 0.0003}, optimizer="sgd")
