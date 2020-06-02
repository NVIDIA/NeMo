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
from nemo.collections.cv.modules.data_layers import CIFAR10DataLayer
from nemo.collections.cv.modules.losses import NLLLoss
from nemo.collections.cv.modules.non_trainables import NonLinearity
from nemo.collections.cv.modules.trainables import ImageEncoder
from nemo.core import DeviceType, NeuralGraph, NeuralModuleFactory, OperationMode, SimpleLossLoggerCallback
from nemo.utils import logging

if __name__ == "__main__":
    # Create the default parser.
    parser = argparse.ArgumentParser(parents=[nm_argparse.NemoArgParser()], conflict_handler='resolve')
    # Parse the arguments
    args = parser.parse_args()

    # Instantiate Neural Factory.
    nf = NeuralModuleFactory(local_rank=args.local_rank, placement=DeviceType.CPU)

    # Data layer - upscale the CIFAR10 images to ImageNet resolution.
    cifar10_dl = CIFAR10DataLayer(height=224, width=224, train=True)
    # The "model".
    image_classifier = ImageEncoder(model_type="resnet50", output_size=10, pretrained=True, name="resnet50")
    nl = NonLinearity(type="logsoftmax", sizes=[-1, 10])
    # Loss.
    nll_loss = NLLLoss()

    # Create a training graph.
    with NeuralGraph(operation_mode=OperationMode.training) as training_graph:
        _, img, tgt = cifar10_dl()
        logits = image_classifier(inputs=img)
        pred = nl(inputs=logits)
        loss = nll_loss(predictions=pred, targets=tgt)
        # Set output - that output will be used for training.
        training_graph.outputs["loss"] = loss

    # Display the graph summmary.
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
