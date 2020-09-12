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

import hydra
from omegaconf import OmegaConf, DictConfig

from nemo.core.classes import ModelPT
from nemo.collections.cv.modules import LeNet5 as LeNet5Module
from nemo.collections.cv.losses import NLLLoss
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import *

import torchvision.transforms as transforms


class MNISTLeNet5(ModelPT):
    """
    The LeNet-5 model.
    """

    def __init__(self):
        super().__init__(cfg=OmegaConf.create())

        # Initialize modules.
        self.module = LeNet5Module()
        self.loss = NLLLoss()

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns:
            :class:`LeNet5Module` input types.
        """
        return self.module.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns:
            :class:`LeNet5Module` output types.
        """
        return self.module.output_types

    @typecheck()
    def forward(self, images):
        """ Propagates data by calling the module :class:`LeNet5Module` forward. """
        return self.module.forward(images=images)

    def training_step(self, batch, what_is_this_input):
        """ Training step, calculate loss. """
        # "Unpack" the batch.
        images, targets = batch

        # Get predictions.
        predictions = self(images=images)

        # Calculate loss.
        loss = self.loss(predictions=predictions, targets=targets)

        # Return it.
        return {"loss": loss}

    @classmethod
    def instantiate_dataloader(
        cls, dataloader_cfg: DictConfig, dataset_cfg: DictConfig, transform_cfg: DictConfig = None
    ):
        """
        Creates the dataset and dataloader.
        Additionally, creates a set of transforms and passes them to dataset.
        """

        if transform_cfg is not None:
            transform = transforms.Compose([hydra.utils.instantiate(trans) for trans in transform_cfg])
        else:
            transform = None
        # Instantiate dataset.
        ds = hydra.utils.instantiate(dataset_cfg, transform=transform)
        # Instantiate dataloader.
        dl = hydra.utils.instantiate(dataloader_cfg, dataset=ds)
        return dl

    def setup_optimization(self, optim_cfg: DictConfig):
        """
        Sets up the optimization (optimizers, schedulers etc.).
        """
        # Instantiate optimizer.
        self._optim = hydra.utils.instantiate(optim_cfg, params=self.parameters())

        # Test of idea - dynamically add of method returning optimizer.
        # Goal: keeping the original "PTL safety mechanisms" when one do not create optimizers =>
        # trainer will act in a standard way.
        def configure_optimizers_dynamic(self):
            return self._optim

        # Add it to the class.
        setattr(self.__class__, 'configure_optimizers', configure_optimizers_dynamic)

    def setup_training_data(self, val_ds_config: Optional[Dict] = None):
        """ Not implemented. """
        pass

    def setup_validation_data(self, val_ds_config: Optional[Dict] = None):
        """ Not implemented. """
        pass

    def save_to(self, save_path: str):
        """ Not implemented. """
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """ Not implemented. """
        pass

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        """ Not implemented. """
        pass

    @classmethod
    def from_pretrained(cls, name: str):
        """ Not implemented. """
        pass
