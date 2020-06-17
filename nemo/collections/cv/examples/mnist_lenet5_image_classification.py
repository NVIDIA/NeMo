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

from torch import optim

import nemo.utils.argparse as nm_argparse
from nemo.collections.cv.modules.data_layers import MNISTDataLayer
from nemo.collections.cv.modules.losses import NLLLoss
from nemo.collections.cv.modules.trainables import LeNet5
from nemo.core import DeviceType, NeuralGraph, NeuralModuleFactory, OperationMode
from nemo.utils import logging

if __name__ == "__main__":
    # Create the default parser.
    parser = argparse.ArgumentParser(parents=[nm_argparse.NemoArgParser()], conflict_handler='resolve')
    # Parse the arguments
    args = parser.parse_args()

    # Instantiate Neural Factory.
    nf = NeuralModuleFactory(local_rank=args.local_rank)

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

    # Display the graph summmary.
    logging.info(training_graph.summary())

    # Create a validation graph, starting from the second data layer.
    with NeuralGraph(operation_mode=OperationMode.evaluation) as evaluation_graph:
        _, x, y, _ = dl_e()
        p = lenet5(images=x)
        loss_e = nll_loss(predictions=p, targets=y)

    # Perform operations on GPU.
    training_graph.to(DeviceType.GPU)
    evaluation_graph.to(DeviceType.GPU)

    # Create optimizer.
    opt = optim.Adam(training_graph.parameters(), lr=0.001)

    # Print frequency.
    freq = 10
    # Train for 5 epochs.
    for epoch in range(5):
        # Configure data loader - once per epoch.
        # Just change the batch_size and turn sample shuffling on.
        training_graph.configure_data_loader(batch_size=128, shuffle=True)

        # Iterate over the whole dataset - in batches.
        for step, batch in enumerate(training_graph.get_batch()):

            # Reset the gradients.
            opt.zero_grad()

            # Forward pass.
            outputs = training_graph.forward(batch)
            # Print loss.
            if step % freq == 0:
                logging.info("Epoch: {} Step: {} Training Loss: {}".format(epoch, step, outputs.loss))

            # Backpropagate the gradients.
            training_graph.backward()

            # Update the parameters.
            opt.step()
        # Epoch ended.

        # Evaluate graph on test set.
        # Configure data loader - once per epoch.
        eval_losses = []
        evaluation_graph.configure_data_loader(batch_size=128)
        # Iterate over the whole dataset - in batches.
        for step, batch in enumerate(evaluation_graph.get_batch()):
            # Forward pass.
            outputs = evaluation_graph.forward(batch)
            eval_losses.append(outputs.loss)
        # Print avg. loss.
        logging.info("Epoch: {} Avg. Evaluation Loss: {}".format(epoch, sum(eval_losses) / len(eval_losses)))
