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
from nemo.core import DeviceType, NeuralModule, NeuralModuleFactory

logging = nemo.logging

# Run on CPU.
nf = NeuralModuleFactory(placement=DeviceType.CPU)

# Instantitate RealFunctionDataLayer defaults to f=torch.sin, sampling from x=[-1, 1]
dl = nemo.tutorials.RealFunctionDataLayer(n=100, f_name="cos", x_lo=-1, x_hi=1, batch_size=128)

# Instantiate a simple feed-forward, single layer neural network.
fx = nemo.tutorials.TaylorNet(dim=4)

# Instantitate loss.
mse_loss = nemo.tutorials.MSELoss()

# Export the model configuration.
fx.export_to_config("/tmp/taylor_net.yml")

# Create a second instance, using the parameters loaded from the previously created configuration.
fx2 = NeuralModule.import_from_config("/tmp/taylor_net.yml")

# Create a graph by connecting the outputs with inputs of modules.
x, y = dl()
# Please note that in the graph are using the "second" instance.
p = fx2(x=x)
loss = mse_loss(predictions=p, target=y)

# SimpleLossLoggerCallback will print loss values to console.
callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[loss], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}')
)

# Invoke the "train" action.
nf.train([loss], callbacks=[callback], optimization_params={"num_epochs": 3, "lr": 0.0003}, optimizer="sgd")
