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

import nemo.utils.argparse as nm_argparse
from nemo.collections.cv.modules.data_layers.cifar10_datalayer import CIFAR10DataLayer
from nemo.collections.cv.modules.losses.nll_loss import NLLLoss
from nemo.collections.cv.modules.non_trainables.reshape_tensor import ReshapeTensor
from nemo.collections.cv.modules.trainables.generic_image_encoder import GenericImageEncoder
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

    # Data layers for training and validation - upscale the CIFAR10 images to ImageNet resolution.
    dl = CIFAR10DataLayer(height=224, width=224, train=True)
    # Model.
    image_encoder = GenericImageEncoder(model_type="vgg16", return_feature_maps=True, pretrained=True, name="vgg16")
    reshaper = ReshapeTensor(input_sizes=[-1, 7, 7, 512], output_sizes=[-1, 25088])
    ffn = FeedForwardNetwork(input_size=25088, output_size=10, hidden_sizes=[1000, 1000], dropout_rate=0.1, final_logsoftmax=True)
    # Loss.
    nll_loss = NLLLoss()

    # 2. Create a training graph.
    with NeuralGraph(operation_mode=OperationMode.training) as training_graph:
        img, tgt = dl()
        feat_map = image_encoder(inputs=img)
        res_img = reshaper(inputs=feat_map)
        pred = ffn(inputs=res_img)
        loss = nll_loss(predictions=pred, targets=tgt)
        # Set output - that output will be used for training.
        training_graph.outputs["loss"] = loss

    # Freeze the pretrained encoder.
    training_graph.freeze()
    logging.info(training_graph.summary())

    # SimpleLossLoggerCallback will print loss values to console.
    callback = SimpleLossLoggerCallback(
        tensors=[loss], print_func=lambda x: logging.info(f'Training Loss: {str(x[0].item())}')
    )

    # Log training metrics to W&B.
    wand_callback = WandbCallback(
        train_tensors=[loss], wandb_name="simple-mnist-fft", wandb_project="cv-collection-image-classification",
    )

    # Invoke the "train" action.
    nf.train(
        training_graph=training_graph,
        callbacks=[callback, wand_callback],
        optimization_params={"num_epochs": 10, "lr": 0.001},
        optimizer="adam",
    )
