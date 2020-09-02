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

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from nemo.collections.cv.datasets import xMNISTDataset, MNISTDatasetConfig
from nemo.collections.cv.losses import NLLLoss
from nemo.collections.cv.modules import LeNet5 as LeNet5Module
from nemo.core.classes import ModelPT
from nemo.core.classes.common import typecheck
from nemo.core.config import Config, CosineAnnealingParams, DataLoaderConfig, NovogradParams
from nemo.core.neural_types import *


@dataclass
class MNISTLeNet5Config(Config):
    """
    Structured config for LeNet-5 model class - that also contains parameters of dataset and dataloader.
    """

    dataset: MNISTDatasetConfig = MNISTDatasetConfig(width=32, height=32)
    dataloader: DataLoaderConfig = DataLoaderConfig(batch_size=64, shuffle=True)
    module: Config = Config()
    # optim: MNISTOptimizer = MNISTOptimizer()


class MNISTLeNet5(ModelPT):
    """
    The LeNet-5 model.
    """

    def __init__(self, cfg: MNISTLeNet5Config = MNISTLeNet5Config(), trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)

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

    def prepare_data(self):
        """ Creates dataset, wrap it with dataloader and return the latter """
        # transform = transforms.Compose([hydra.utils.instantiate(trans) for trans in self.data.tf])
        # self.train_set = hydra.utils.instantiate(self.data.ds, transform=transform, train=True)
        # Instantiate dataset.
        self._train_set = xMNISTDataset(self._cfg.training.dataset.params)

    def training_step(self, batch, what_is_this_input):
        """ Training step, calculate loss. """
        # "Unpack" the batch.
        _, images, targets, _ = batch

        # Get predictions.
        predictions = self(images=images)

        # Calculate loss.
        loss = self.loss(predictions=predictions, targets=targets)

        # Return it.
        return {"loss": loss}

    def train_dataloader(self):
        """ Return the training dataloader. """
        return hydra.utils.instantiate(self._cfg.training.dataloader, dataset=self._train_set)

    def configure_optimizers(self):
        """
        Sets the optimizer.
        """
        optimizer = hydra.utils.instantiate(self._cfg.optim, params=self.parameters())
        return optimizer

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
