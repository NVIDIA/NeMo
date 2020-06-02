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
from nemo.collections.cv.modules.data_layers import CIFAR100DataLayer
from nemo.collections.cv.modules.losses import NLLLoss
from nemo.collections.cv.modules.non_trainables import NonLinearity, ReshapeTensor
from nemo.collections.cv.modules.trainables import FeedForwardNetwork, ImageEncoder
from nemo.core import DeviceType, NeuralGraph, NeuralModuleFactory, OperationMode, SimpleLossLoggerCallback
from nemo.utils import logging

if __name__ == "__main__":
    # Create the default parser.
    parser = argparse.ArgumentParser(parents=[nm_argparse.NemoArgParser()], conflict_handler='resolve')
    # Parse the arguments
    args = parser.parse_args()

    # Instantiate Neural Factory.
    nf = NeuralModuleFactory(local_rank=args.local_rank, placement=DeviceType.CPU)

    # Data layer - upscale the CIFAR100 images to ImageNet resolution.
    cifar100_dl = CIFAR100DataLayer(height=224, width=224, train=True)
    # The "model".
    image_encoder = ImageEncoder(model_type="vgg16", return_feature_maps=True, pretrained=True, name="vgg16")
    reshaper = ReshapeTensor(input_sizes=[-1, 7, 7, 512], output_sizes=[-1, 25088])
    ffn = FeedForwardNetwork(input_size=25088, output_size=100, hidden_sizes=[1000, 1000], dropout_rate=0.1)
    nl = NonLinearity(type="logsoftmax", sizes=[-1, 100])
    # Loss.
    nll_loss = NLLLoss()

    # Create a training graph.
    with NeuralGraph(operation_mode=OperationMode.training) as training_graph:
        _, img, _, _, fine_target, _ = cifar100_dl()
        feat_map = image_encoder(inputs=img)
        res_img = reshaper(inputs=feat_map)
        logits = ffn(inputs=res_img)
        pred = nl(inputs=logits)
        loss = nll_loss(predictions=pred, targets=fine_target)
        # Set output - that output will be used for training.
        training_graph.outputs["loss"] = loss

    # Freeze the pretrained encoder.
    training_graph.freeze(["vgg16"])
    logging.info(training_graph.summary())

    # SimpleLossLoggerCallback will print loss values to console.
    callback = SimpleLossLoggerCallback(
        tensors=[loss], print_func=lambda x: logging.info(f'Training Loss: {str(x[0].item())}')
    )

    # Invoke the "train" action.
    nf.train(
        training_graph=training_graph,
        callbacks=[callback],
        optimization_params={"num_epochs": 10, "lr": 0.001},
        optimizer="adam",
    )
