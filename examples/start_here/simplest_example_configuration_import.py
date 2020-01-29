# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2019 NVIDIA. All Rights Reserved.
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
from nemo.core import DeviceType

# Run on CPU.
nf = nemo.core.NeuralModuleFactory(placement=DeviceType.CPU)


# instantiate necessary neural modules
# RealFunctionDataLayer defaults to f=torch.sin, sampling from x=[-4, 4]
# dl = nemo.tutorials.RealFunctionDataLayer(n=10000, f_name="cos", x=[-4, 4], batch_size=128)
dl = nemo.tutorials.RealFunctionDataLayer(n=100, f_name="cos", x_lo=-1, x_hi=1, batch_size=128)


fx = nemo.tutorials.TaylorNet(dim=4)
loss = nemo.tutorials.MSELoss()

# describe activation's flow
x, y = dl()
p = fx(x=x)
lss = loss(predictions=p, target=y)

# SimpleLossLoggerCallback will print loss values to console.
callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[lss], print_func=lambda x: nemo.logging.info(f'Train Loss: {str(x[0].item())}')
)


# Invoke "train" action
nf.train([lss], callbacks=[callback], optimization_params={"num_epochs": 3, "lr": 0.0003}, optimizer="sgd")
