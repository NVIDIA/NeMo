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

import argparse

from torch import max, mean, stack, tensor

import nemo.utils.argparse as nm_argparse
from nemo.collections.cv.modules.data_layers import MNISTDataLayer
from nemo.collections.cv.modules.losses import NLLLoss
from nemo.collections.cv.modules.trainables import LeNet5
from nemo.core import (
    DeviceType,
    EvaluatorCallback,
    NeuralGraph,
    NeuralModuleFactory,
    OperationMode,
    SimpleLossLoggerCallback,
)
from nemo.utils import logging

if __name__ == "__main__":
    # Create the default parser.
    parser = argparse.ArgumentParser(parents=[nm_argparse.NemoArgParser()], conflict_handler='resolve')
    # Parse the arguments
    args = parser.parse_args()

    # Instantiate Neural Factory.
    nf = NeuralModuleFactory(local_rank=args.local_rank, placement=DeviceType.GPU)

    # Data layers for training and validation.
    dl = MNISTDataLayer(height=32, width=32, train=True)
    dl_e = MNISTDataLayer(height=32, width=32, train=False)
    # The "model".
    lenet5 = LeNet5()
    # Loss.
    nll_loss = NLLLoss()

    # Create a training graph.
    with NeuralGraph(operation_mode=OperationMode.training) as training_graph:
        _, x, y, _ = dl()
        p = lenet5(images=x)
        loss = nll_loss(predictions=p, targets=y)
        # Set output - that output will be used for training.
        training_graph.outputs["loss"] = loss

    # Display the graph summmary.
    logging.info(training_graph.summary())

    # Create a validation graph, starting from the second data layer.
    with NeuralGraph(operation_mode=OperationMode.evaluation) as evaluation_graph:
        _, x, y, _ = dl_e()
        p = lenet5(images=x)
        loss_e = nll_loss(predictions=p, targets=y)

    # Display the graph summmary.
    logging.info(evaluation_graph.summary())

    # Create the callbacks.
    def eval_loss_per_batch_callback(tensors, global_vars):
        if "eval_loss" not in global_vars.keys():
            global_vars["eval_loss"] = []
        for key, value in tensors.items():
            if key.startswith("loss"):
                global_vars["eval_loss"].append(mean(stack(value)))

    def eval_loss_epoch_finished_callback(global_vars):
        eloss = max(tensor(global_vars["eval_loss"]))
        logging.info("Evaluation Loss: {0}".format(eloss))
        return dict({"Evaluation Loss": eloss})

    ecallback = EvaluatorCallback(
        eval_tensors=[loss_e],
        user_iter_callback=eval_loss_per_batch_callback,
        user_epochs_done_callback=eval_loss_epoch_finished_callback,
        eval_step=100,
    )

    # SimpleLossLoggerCallback will print loss values to console.
    callback = SimpleLossLoggerCallback(
        tensors=[loss], print_func=lambda x: logging.info(f'Training Loss: {str(x[0].item())}')
    )

    # Invoke the "train" action.
    nf.train(
        training_graph=training_graph,
        callbacks=[callback, ecallback],
        optimization_params={"num_epochs": 10, "lr": 0.001},
        optimizer="adam",
    )
