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

from nemo.backends.pytorch.tutorials import MSELoss, RealFunctionDataLayer, TaylorNet
from nemo.core import DeviceType, NeuralGraph, NeuralModuleFactory, OperationMode, SimpleLossLoggerCallback
from nemo.utils import logging, AppState

nf = NeuralModuleFactory(placement=DeviceType.CPU)
# Instantiate the necessary neural modules.
dl = RealFunctionDataLayer(n=100, batch_size=32, name="dl")
m1 = TaylorNet(dim=4, name="m1")
m2 = TaylorNet(dim=4, name="m2")
loss = MSELoss(name="loss")

logging.info(
    "This example shows how one can nest one graph into another - with manual binding of selected output ports."
    F" Please note that the nested graph can be used exatly like any other module."
)

with NeuralGraph(operation_mode=OperationMode.training, name="g1") as g1:
    x, t = dl()
    prediction1 = m1(x=x)
    prediction2 = m2(x=x)
    # Manually bind the selected outputs.
    g1.outputs["ix"] = x
    g1.outputs["te"] = t
    g1.outputs["prediction"] = prediction2

# Serialize graph
serialized_g1 = g1.serialize()
print("Serialized:\n", serialized_g1)

# Delete everything!
del g1
del dl
del m1
del m2

print(AppState().modules.summary())
assert len(AppState().modules) == 1 # only the "loss" module

# Deserialize graph.
g1_copy = NeuralGraph.deserialize(serialized_g1, reuse_existing_modules=False, name="g1_copy")
serialized_g1_copy = g1_copy.serialize()
print("Deserialized:\n", serialized_g1_copy)


with NeuralGraph(operation_mode=OperationMode.training, name="g1.1") as g2:
    x1, t1, p1 = g1_copy()
    lss = loss(predictions=p1, target=t1)
    # Manual output.
    g2.outputs["loss"] = lss


# SimpleLossLoggerCallback will print loss values to console.
callback = SimpleLossLoggerCallback(
    tensors=[g2.output_tensors["loss"]], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}'),
)

# Invoke "train" action.
nf.train(
    [g2.output_tensors["loss"]],
    callbacks=[callback],
    optimization_params={"num_epochs": 2, "lr": 0.0003},
    optimizer="sgd",
)
