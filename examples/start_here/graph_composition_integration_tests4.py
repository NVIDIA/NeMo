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

import nemo
from nemo.core import NeuralGraph, OperationMode

logging = nemo.logging

nf = nemo.core.NeuralModuleFactory()
# Instantiate the necessary neural modules.
dl_training = nemo.tutorials.RealFunctionDataLayer(n=10000, batch_size=128)
dl_validation = nemo.tutorials.RealFunctionDataLayer(n=10000, batch_size=128)
fx = nemo.tutorials.TaylorNet(dim=4)
loss = nemo.tutorials.MSELoss()

logging.info(
    "This example shows how one can nest one graph (representing the our trained model) into"
    F" training and validation graphs."
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

with NeuralGraph(operation_mode=OperationMode.inference, name="validation_graph") as validation_graph:
    # Take outputs from the training DL.
    x_valid, t_valid = dl_training()
    # Pass them to the trainable module.
    p_valid = trainable_module(x=x_valid)
    # Pass both of them to loss.
    loss_e = loss(predictions=p_valid, target=t_valid)


# Callbacks to print info to console and Tensorboard.
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[lss], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}')
)


def batch_loss_per_batch_callback(tensors, global_vars):
    if "batch_loss" not in global_vars.keys():
        global_vars["batch_loss"] = []
    for key, value in tensors.items():
        if key.startswith("loss"):
            global_vars["batch_loss"].append(torch.mean(torch.stack(value)))


def batch_loss_epoch_finished_callback(global_vars):
    epoch_loss = torch.max(torch.tensor(global_vars["batch_loss"]))
    print("Evaluation Loss: {0}".format(epoch_loss))
    return dict({"Evaluation Loss": epoch_loss})


eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=[loss_e],
    user_iter_callback=batch_loss_per_batch_callback,
    user_epochs_done_callback=batch_loss_epoch_finished_callback,
    eval_step=100,
)

# Invoke "train" action.
nf.train(
    [lss],
    callbacks=[train_callback, eval_callback],
    optimization_params={"num_epochs": 3, "lr": 0.0003},
    optimizer="sgd",
)
