# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict, Optional

from omegaconf import OmegaConf
from torch import nn

from nemo.collections.cv.losses import NLLLoss
from nemo.collections.cv.models.model import Model
from nemo.collections.cv.modules import ImageEncoder, ImageEncoderConfig
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import NeuralType


class ResNet50(Model):
    """
    The LeNet-5 model.
    """

    def __init__(self, cfg: ImageEncoderConfig):
        super().__init__(cfg=OmegaConf.create())

        # Initialize modules.
        self.classifier = ImageEncoder(**cfg)
        self.loss = NLLLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns:
            :class:`ImageEncoder` input types.
        """
        return {"images": self.classifier.input_types["inputs"]}

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns:
            :class:`ImageEncoder` output types.
        """
        return self.classifier.output_types

    @typecheck()
    def forward(self, images):
        """ Propagates data by calling the module :class:`ResNet50` forward, calculates and returns loss. """
        return self.classifier.forward(inputs=images)

    def training_step(self, batch, what_is_this_input):
        """ Training step, calculate loss. """
        # "Unpack" the batch.
        images, targets = batch

        # Get logigs.
        logits = self.classifier(inputs=images)

        # Apply log softmax on logits.
        log_sm = self.log_softmax(logits)

        # Calculate loss.
        loss = self.loss(predictions=log_sm, targets=targets)

        # Return it.
        return {"loss": loss}
