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

from dataclasses import dataclass
from typing import Dict, Optional

from torch.utils.data import DataLoader
from torch.optim import Optimizer

from nemo.core.neural_types import *
from nemo.core.classes import ModelPT
from nemo.core.classes.common import typecheck

from nemo.core.config import Config, DataLoaderConfig, NovogradConfig
from nemo.core.optim.optimizers import get_optimizer

from nemo.collections.cv.losses import NLLLoss
from nemo.collections.cv.datasets import MNISTDataset, MNISTDatasetConfig
from nemo.collections.cv.modules import LeNet5 as LeNet5Module

from nemo.utils.decorators import experimental

__all__ = ['MNISTLeNet5', 'MNISTLeNet5Config']


@dataclass
class MNISTLeNet5Config(Config):
    """
    Structured config for LeNet-5 model class - that also contains parameters of dataset and dataloader.

    (This example shows that we can inherit from other configs.)

    Args:
        opt: Lets use Novograd optimizer.
    """
    opt: NovogradConfig = NovogradConfig()
    dataset: MNISTDatasetConfig = MNISTDatasetConfig(width=32, height=32)
    dataloader: DataLoaderConfig = DataLoaderConfig(batch_size=64, shuffle=True)


@experimental
class MNISTLeNet5(ModelPT):
    def __init__(self, cfg: MNISTLeNet5Config = MNISTLeNet5Config()):
        super().__init__()
        # Remember config - should be moved to base init.
        self._cfg = cfg

        # Initialize modules.
        self.module = LeNet5Module()
        self.loss = NLLLoss()

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.module.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.module.output_types

    @typecheck()
    def forward(self, images):
        return self.module.forward(images=images)

    def configure_optimizers(self):
        # Get optimizer class.
        optimizer_cls = get_optimizer(self._cfg.opt.cls)
        # Instantiate the optimizer, set model parameters and pass other kwargs.
        optimizer = optimizer_cls(params=self.parameters(), **self._cfg.opt.params)
        return optimizer

    def training_step(self, batch, what_is_this_input):
        # "Unpack" the batch.
        _, images, targets, _ = batch

        # Get predictions.
        predictions = self(images=images)

        # Calculate loss.
        loss = self.loss(predictions=predictions, targets=targets)

        # Return it.
        return {"loss": loss}
        # of course "return loss" doesn't work :]

    def train_dataloader(self):
        """ Create dataset, wrap it with dataloader and return the latter """
        # Instantiate dataset.
        mnist_ds = MNISTDataset(self._cfg.dataset)
        # Configure data loader.
        train_dataloader = DataLoader(dataset=mnist_ds, **(self._cfg.dataloader))

        return train_dataloader

    def setup_training_data(self, train_data_layer_params: Optional[Dict]):
        """ Dummy methods """
        pass

    def setup_validation_data(self, val_data_layer_params: Optional[Dict]):
        """ Dummy methods """
        pass

    def setup_test_data(self, test_data_layer_params: Optional[Dict]):
        """ Dummy methods """
        pass

    def setup_optimization(self, optim_params: Optional[Dict] = None) -> Optimizer:
        """ Dummy methods """
        pass

    def save_to(self, save_path: str):
        """ Why do I need that in experimental module? """
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """ Why do I need that in experimental module? """
        pass

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        """ Why do I need that in experimental module? """
        pass

    @classmethod
    def from_pretrained(cls, name: str):
        """ Why do I need that in experimental module? """
        pass

    def export(self, **kwargs):
        """ Why do I need that in experimental module? """
        pass
