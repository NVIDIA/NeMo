# Copyright (C) tkornuta, NVIDIA AI Applications Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Tomasz Kornuta"

import math
import torch

import nemo
import torch

from nemo.core import NeuralType, DeviceType

from nemo_cv.modules.mnist_datalayer import MNISTDataLayer
from nemo_cv.modules.lenet5 import LeNet5
from nemo_cv.modules.nll_loss import NLLLoss


# 0. Instantiate Neural Factory with supported backend
nf = nemo.core.NeuralModuleFactory(placement=DeviceType.GPU)


#############################################################################
# 1. Instantiate necessary neural modules
dl = MNISTDataLayer(
    batch_size=64,
    data_folder="~/data/mnist",
    train=True,
    shuffle=True
)
lenet5 = LeNet5()
nll_loss = NLLLoss()

# Data layer for the validation.
dl_e = MNISTDataLayer(
    batch_size=64,
    data_folder="~/data/mnist",
    train=False,
    shuffle=True
)


#############################################################################
# 2. Describe activation's flow
x, y = dl()
p = lenet5(images=x)
loss = nll_loss(predictions=p, targets=y)


#############################################################################
# Create validation graph, starting from the second data layer.
x, y = dl_e()
p = lenet5(images=x)
nll_loss_e = NLLLoss()
loss_e = nll_loss_e(predictions=p, targets=y)


def eval_loss_per_batch_callback(tensors, global_vars):
    if "eval_loss" not in global_vars.keys():
        global_vars["eval_loss"] = []
    for key, value in tensors.items():
        if key.startswith("loss"):
            global_vars["eval_loss"].append(torch.mean(torch.stack(value)))


def eval_loss_epoch_finished_callback(global_vars):
    eloss = torch.max(torch.tensor(global_vars["eval_loss"]))
    print("Evaluation Loss: {0}".format(eloss))
    return dict({"Evaluation Loss": eloss})


ecallback = nemo.core.EvaluatorCallback(
    eval_tensors=[loss_e],
    user_iter_callback=eval_loss_per_batch_callback,
    user_epochs_done_callback=eval_loss_epoch_finished_callback,
    eval_step=100)


# SimpleLossLoggerCallback will print loss values to console.
callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[loss],
    print_func=lambda x: print(f'Train Loss: {str(x[0].item())}'))


# Invoke "train" action
nf.train([loss], callbacks=[callback, ecallback],
         optimization_params={"num_epochs": 10, "lr": 0.001},
         optimizer="adam")
