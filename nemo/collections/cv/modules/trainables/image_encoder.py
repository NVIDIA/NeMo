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
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2019
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
# =============================================================================

__author__ = "Tomasz Kornuta"

"""
This file contains code artifacts adapted from the original implementation:
https://github.com/IBM/pytorchpipe/blob/develop/ptp/components/models/vision/image_encoder.py
"""

from typing import Optional

import torch
import torchvision.models as models

from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_types import AxisKind, AxisType, ImageFeatureValue, ImageValue, LogitsType, NeuralType
from nemo.utils import logging
from nemo.utils.configuration_parsing import get_value_from_dictionary
from nemo.utils.decorators import add_port_docs

__all__ = ['ImageEncoder']


class ImageEncoder(TrainableNM):
    """
    Neural Module implementing a general-usage image encoderds.
    It encapsulates several models from TorchVision (VGG16, ResNet152 and DensNet121, naming a few).
    Offers two operation modes and can return: image embeddings vs feature maps.
    """

    def __init__(
        self,
        model_type: str,
        output_size: Optional[int] = None,
        return_feature_maps: bool = False,
        pretrained: bool = False,
        name: Optional[str] = None,
    ):
        """
        Initializes the ``ImageEncoder`` model, creates the required "backbone".

        Args:
            model_type: Type of backbone (Options: VGG16 | DenseNet121 | ResNet152 | ResNet50)
            output_size: Size of the output layer (Optional, Default: None)
            return_feature_maps: Return mode: image embeddings vs feature maps (Default: False)
            pretrained: Loads pretrained model (Default: False)
            name: Name of the module (DEFAULT: None)
        """
        TrainableNM.__init__(self, name=name)

        # Get operation modes.
        self._return_feature_maps = return_feature_maps

        # Get model type.
        self._model_type = get_value_from_dictionary(
            model_type, "vgg16 | densenet121 | resnet152 | resnet50".split(" | ")
        )

        # Get output size (optional - not in feature_maps).
        self._output_size = output_size

        if self._model_type == 'vgg16':
            # Get VGG16
            self._model = models.vgg16(pretrained=pretrained)

            if self._return_feature_maps:
                # Use only the "feature encoder".
                self._model = self._model.features

                # Remember the output feature map dims.
                self._feature_map_height = 7
                self._feature_map_width = 7
                self._feature_map_depth = 512

            else:
                # Use the whole model, but "reshape"/reinstantiate the last layer ("FC6").
                self._model.classifier._modules['6'] = torch.nn.Linear(4096, self._output_size)

        elif self._model_type == 'densenet121':
            # Get densenet121
            self._model = models.densenet121(pretrained=pretrained)

            if self._return_feature_maps:
                raise ConfigurationError("'densenet121' doesn't support 'return_feature_maps' mode (yet)")

            # Use the whole model, but "reshape"/reinstantiate the last layer ("FC6").
            self._model.classifier = torch.nn.Linear(1024, self._output_size)

        elif self._model_type == 'resnet152':
            # Get resnet152
            self._model = models.resnet152(pretrained=pretrained)

            if self._return_feature_maps:
                # Get all modules exluding last (avgpool) and (fc)
                modules = list(self._model.children())[:-2]
                self._model = torch.nn.Sequential(*modules)

                # Remember the output feature map dims.
                self._feature_map_height = 7
                self._feature_map_width = 7
                self._feature_map_depth = 2048

            else:
                # Use the whole model, but "reshape"/reinstantiate the last layer ("FC6").
                self._model.fc = torch.nn.Linear(2048, self._output_size)

        elif self._model_type == 'resnet50':
            # Get resnet50
            self._model = models.resnet50(pretrained=pretrained)

            if self._return_feature_maps:
                # Get all modules exluding last (avgpool) and (fc)
                modules = list(self._model.children())[:-2]
                self._model = torch.nn.Sequential(*modules)

                # Remember the output feature map dims.
                self._feature_map_height = 7
                self._feature_map_width = 7
                self._feature_map_depth = 2048

            else:
                # Use the whole model, but "reshape"/reinstantiate the last layer ("FC6").
                self._model.fc = torch.nn.Linear(2048, self._output_size)

    @property
    @add_port_docs()
    def input_ports(self):
        """
        Returns definitions of module input ports.
        """
        return {
            "inputs": NeuralType(
                axes=(
                    AxisType(kind=AxisKind.Batch),
                    AxisType(kind=AxisKind.Channel, size=3),
                    AxisType(kind=AxisKind.Height, size=224),
                    AxisType(kind=AxisKind.Width, size=224),
                ),
                elements_type=ImageValue(),
                # TODO: actually encoders pretrained on ImageNet require special image normalization.
                # Probably this should be a new image type.
            )
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """
        Returns definitions of module output ports.
        """
        # Return neural type.
        if self._return_feature_maps:
            return {
                "outputs": NeuralType(
                    axes=(
                        AxisType(kind=AxisKind.Batch),
                        AxisType(kind=AxisKind.Channel, size=self._feature_map_depth),
                        AxisType(kind=AxisKind.Height, size=self._feature_map_height),
                        AxisType(kind=AxisKind.Width, size=self._feature_map_width),
                    ),
                    elements_type=ImageFeatureValue(),
                )
            }
        else:
            return {
                "outputs": NeuralType(
                    axes=(AxisType(kind=AxisKind.Batch), AxisType(kind=AxisKind.Any, size=self._output_size),),
                    elements_type=LogitsType(),
                )
            }

    def forward(self, inputs):
        """
        Main forward pass of the model.

        Args:
            inputs: expected stream containing images [BATCH_SIZE x IMAGE_DEPTH x IMAGE_HEIGHT x IMAGE WIDTH]
        
        Returns:
            outpus: added stream containing outputs [BATCH_SIZE x OUTPUT_SIZE]
                OR [BATCH_SIZE x OUTPUT_DEPTH x OUTPUT_HEIGHT x OUTPUT_WIDTH]
        """
        # print("{}: input shape: {}, device: {}\n".format(self.name, inputs.shape, inputs.device))

        outputs = self._model(inputs)

        # Add outputs to datadict.
        return outputs
