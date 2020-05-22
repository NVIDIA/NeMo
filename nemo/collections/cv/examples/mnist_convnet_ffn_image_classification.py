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
from copy import deepcopy

import numpy as np
from torch import max, mean, stack, tensor

import nemo.utils.argparse as nm_argparse
from nemo.backends import get_state_dict
from nemo.collections.cv.modules.data_layers.mnist_datalayer import MNISTDataLayer
from nemo.collections.cv.modules.losses.nll_loss import NLLLoss
from nemo.collections.cv.modules.non_trainables.reshape_tensor import ReshapeTensor
from nemo.collections.cv.modules.trainables.convnet_encoder import ConvNetEncoder
from nemo.collections.cv.modules.trainables.feed_forward_network import FeedForwardNetwork
from nemo.core import (
    DeviceType,
    NeuralGraph,
    NeuralModuleFactory,
    OperationMode,
    SimpleLossLoggerCallback,
    WandbCallback,
)
from nemo.utils import logging

if __name__ == "__main__":
    # Create the default parser.
    parser = argparse.ArgumentParser(parents=[nm_argparse.NemoArgParser()], conflict_handler='resolve')
    # Parse the arguments
    args = parser.parse_args()

    # 0. Instantiate Neural Factory.
    nf = NeuralModuleFactory(local_rank=args.local_rank, placement=DeviceType.CPU)

    # Data layers for training and validation.
    dl = MNISTDataLayer(height=28, width=28, train=True)
    # Model.
    cnn = ConvNetEncoder(input_depth=1, input_height=28, input_width=28)
    reshaper = ReshapeTensor(input_dims=[-1, 16, 1, 1], output_dims=[-1, 16])
    ffn = FeedForwardNetwork(input_size=16, output_size=10, dropout_rate=0.1, final_logsoftmax=True)
    # Loss.
    nll_loss = NLLLoss()

    # 2. Create a training graph.
    with NeuralGraph(operation_mode=OperationMode.training) as training_graph:
        img, tgt = dl()
        feat_map = cnn(inputs=img)
        res_img = reshaper(inputs=feat_map)
        pred = ffn(inputs=res_img)
        loss = nll_loss(predictions=pred, targets=tgt)
        # Set output - that output will be used for training.
        training_graph.outputs["loss"] = loss

    # SimpleLossLoggerCallback will print loss values to console.
    callback = SimpleLossLoggerCallback(
        tensors=[loss], print_func=lambda x: logging.info(f'Training Loss: {str(x[0].item())}')
    )

    # Log training metrics to W&B.
    # wand_callback = WandbCallback(
    #    train_tensors=[loss],
    #    wandb_name="simple-mnist-fft",
    #    wandb_project="cv-collection-image-classification",
    # )

    # Get CNN weights before training.
    weights = deepcopy(get_state_dict(cnn)["_conv3.bias"]).numpy()

    # Invoke the "train" action.
    nf.train(
        training_graph=training_graph,
        callbacks=[callback],
        optimization_params={"max_steps": 100, "lr": 0.001},
        optimizer="adam",
    )

    # Get CNN weights after training.
    weights2 = deepcopy(get_state_dict(cnn)["_conv3.bias"]).numpy()

    logging.info("Before training:\n{}".format(weights))
    logging.info("After training:\n{}".format(weights2))

    if np.array_equal(weights, weights2):
        logging.error("Module weights not updated during training")
