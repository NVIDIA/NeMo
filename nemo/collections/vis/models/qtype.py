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

from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field, MISSING

import hydra
from omegaconf import OmegaConf, DictConfig

from nemo.collections.cv.models.model import Model
from nemo.collections.cv.losses import NLLLoss
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import *

from nemo.collections.vis.transforms import Compose


@dataclass
class QTypeConfig:
    """
    Structured config for the QType model.

    For more details please refer to:
    https://cs.stanford.edu/people/jcjohns/clevr/

    Args:
        _target_: Specification of target class
    """

    # Target class name.
    _target_: str = "nemo.collections.vis.models.QType"
    # text_transforms: List[TextTransforms]
    text_transforms: Optional[Any] = None  # List[Any] = field(default_factory=list) ?


class QType(Model):
    """
    The LeNet-5 model.
    """

    def __init__(self, cfg: QTypeConfig = QTypeConfig()):
        super().__init__(cfg=cfg)

        # Check transforms configuration.
        if cfg.text_transforms is not None:
            self.transforms = Compose([hydra.utils.instantiate(trans) for trans in cfg.text_transforms])
        else:
            self.transforms = None

    def forward(self, sentences):
        """ Propagates data throught the model. """

        if self.transforms is not None:
            sentences = self.transforms(sentences)

        return sentences

    def training_step(self, batch, what_is_this_input):
        """ Training step, calculate loss. """
        # "Unpack" the batch.
        # sentence, targets = batch

        # Get predictions.
        # predictions = self(images=images)

        # Calculate loss.
        # loss = self.loss(predictions=predictions, targets=targets)

        # Return it.
        # return {"loss": loss}
        return 1
