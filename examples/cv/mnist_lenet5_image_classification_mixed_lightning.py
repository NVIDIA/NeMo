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

import pytorch_lightning as ptl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from nemo.collections.cv.datasets import MNISTDataset, MNISTDatasetConfig
from nemo.collections.cv.models import LeNet5, LeNet5Config
from nemo.core.config import Config, DataLoaderConfig, TrainerConfig, set_config
from nemo.utils import logging


@dataclass
class AppConfig(Config):
    """
    This is structured config for this application.

    Args:
        name: Description of the application.
        dataset: contains configuration of dataset.
        dataloader: contains configuration of dataloader.
        trainer: configuration of the trainer.
        model: configuation of the model.
    """

    name: str = "Training of a LeNet-5 Model using a mixed PyTorch - PyTorchLightning approach."
    dataset: MNISTDatasetConfig = MNISTDatasetConfig(width=32, height=32)
    dataloader: DataLoaderConfig = DataLoaderConfig(batch_size=128, shuffle=True)
    trainer: TrainerConfig = TrainerConfig()
    model: LeNet5Config = LeNet5Config()


@set_config(config=AppConfig)
def main(cfg: DictConfig):

    # Show configuration - user can modify every parameter from command line!
    print("=" * 80 + " Hydra says hello! " + "=" * 80)
    print(cfg.pretty())

    # The "model".
    lenet5 = LeNet5(cfg.model)

    # Instantiate dataset.
    mnist_ds = MNISTDataset(cfg.dataset)
    # Configure data loader.
    train_dataloader = DataLoader(dataset=mnist_ds, **(cfg.dataloader))

    # Create trainer.
    trainer = ptl.Trainer(**(cfg.trainer))

    # Train.
    trainer.fit(model=lenet5, train_dataloader=train_dataloader)


if __name__ == "__main__":
    main()
