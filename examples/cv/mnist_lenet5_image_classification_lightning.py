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
from torch.utils.data import DataLoader

from nemo.collections.cv.datasets import MNISTDataset
from nemo.collections.cv.losses import NLLLoss
from nemo.collections.cv.modules import LeNet5
from nemo.utils import logging

if __name__ == "__main__":

    # Dataset.
    mnist_ds = MNISTDataset(height=32, width=32, train=True)
    # The "model".
    lenet5 = LeNet5()
    # Loss.
    nll_loss = NLLLoss()

    # Create a validation graph, starting from the second data layer.
    #with NeuralGraph(operation_mode=OperationMode.evaluation) as evaluation_graph:
    #    _, x, y, _ = dl_e()
    #    p = lenet5(images=x)
    #    loss_e = nll_loss(predictions=p, targets=y)

    # Create optimizer.
    opt = optim.Adam(lenet5.parameters(), lr=0.001)

    # Print frequency.
    freq = 10

    # Configure data loader
    dl = DataLoader(dataset=mnist_ds, batch_size=128, shuffle=True)

    # Iterate over the whole dataset - in batches.
    for step, (_, images, targets, _) in enumerate(dl):

        # Reset the gradients.
        opt.zero_grad()

        # Forward pass.
        predictions = lenet5(images=images)

        # Calculate loss.
        loss = nll_loss(predictions=predictions, targets=targets)

        # Print loss.
        if step % freq == 0:
            logging.info("Step: {} Training Loss: {}".format(step, loss))

        # Backpropagate the gradients.
        loss.backward()

        # Update the parameters.
        opt.step()
    # Epoch ended.
