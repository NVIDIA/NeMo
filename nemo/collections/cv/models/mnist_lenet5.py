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
from typing import Any, Dict, Optional

from torch.optim import Optimizer
from torch.utils.data import DataLoader

from nemo.collections.cv.datasets import MNISTDataset, MNISTDatasetConfig
from nemo.collections.cv.losses import NLLLoss
from nemo.collections.cv.modules import LeNet5 as LeNet5Module
from nemo.core.classes import ModelPT
from nemo.core.classes.common import typecheck
from nemo.core.config import Config, CosineAnnealingParams, DataLoaderConfig, NovogradParams
from nemo.core.neural_types import *
from nemo.core.optim.lr_scheduler import prepare_lr_scheduler
from nemo.utils.decorators import experimental


@dataclass
class NovogradScheduler(Config):
    """
    Scheduler setup for novograd
    """

    # Mandatory parameters
    name: str = "CosineAnnealing"
    args: CosineAnnealingParams = CosineAnnealingParams()  # can be a string to param, or "auto", just like in YAML

    # pytorch lightning parameters
    monitor: str = "val_loss"
    iters_per_batch: Optional[float] = None  # computed at runtime
    max_steps: Optional[int] = None  # computed at runtime or explicitly set here
    reduce_on_plateau: bool = False


@dataclass
class MNISTOptimizer(Config):
    """
    Optimizer setup for novograd
    """

    # Mandatory arguments
    name: str = "novograd"
    lr: float = 0.01

    args: NovogradParams = NovogradParams(betas=(0.8, 0.5))
    sched: NovogradScheduler = NovogradScheduler()


@dataclass
class MNISTLeNet5Config(Config):
    """
    Structured config for LeNet-5 model class - that also contains parameters of dataset and dataloader.

    (This example shows that we can inherit from other configs.)

    Args:
        dataset: MNIST dataset config.
        dataloader: Dataloader config.
        module: module config (default one).
        optim: Optimizer + Scheduler config.
    """

    dataset: MNISTDatasetConfig = MNISTDatasetConfig(width=32, height=32)
    dataloader: DataLoaderConfig = DataLoaderConfig(batch_size=64, shuffle=True)
    module: Config = Config()
    optim: MNISTOptimizer = MNISTOptimizer()


@experimental
class MNISTLeNet5(ModelPT):
    def __init__(self, cfg: MNISTLeNet5Config = MNISTLeNet5Config()):
        super().__init__()
        # Remember config - should be moved to base init.
        self._cfg = cfg

        # Initialize modules.
        self.module = LeNet5Module(cfg.module)
        self.loss = NLLLoss()

        # This will be set by setup_training_data
        self._train_dl = None
        # This will be set by setup_validation_data
        self._val_dl = None
        # This will be set by setup_test_data
        self._test_dl = None

        self._optimizer = None
        self._scheduler = None

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.module.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.module.output_types

    @typecheck()
    def forward(self, images):
        return self.module.forward(images=images)

    # This will fail as train_dataloader is not yet instantiated !
    # def configure_optimizers(self):
    #     # Get optimizer class.
    #     optim_config = self._cfg.optim
    #     optimizer = self.setup_optimization(optim_config)
    #     scheduler = prepare_lr_scheduler(
    #         optimizer, scheduler_config=optim_config, train_dataloader=self._train_dataloader
    #     )
    #     return [optimizer], [scheduler]

    def setup_optimization(self, optim_params=None) -> Optimizer:
        optim_params = optim_params or self._cfg.optim
        self._optimizer = super().setup_optimization(optim_params)
        self._scheduler = prepare_lr_scheduler(
            optimizer=self._optimizer, scheduler_config=optim_params, train_dataloader=self._train_dl
        )

    def setup_training_data(self, train_data_layer_params: Optional[Dict] = None):
        """ Create dataset, wrap it with dataloader and return the latter """
        # Instantiate dataset.
        mnist_ds = MNISTDataset(self._cfg.dataset)
        # Configure data loader.
        train_dataloader = DataLoader(dataset=mnist_ds, **(self._cfg.dataloader))
        self._train_dl = train_dataloader

    def setup_validation_data(self, val_data_layer_params: Optional[Dict] = None):
        self._val_dl = None

    def setup_test_data(self, test_data_layer_params: Optional[Dict] = None):
        self._test_dl = None

    def configure_optimizers(self):
        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]

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
        return self._train_dl

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass

    @classmethod
    def from_pretrained(cls, name: str):
        pass

    def export(self, **kwargs):
        pass
