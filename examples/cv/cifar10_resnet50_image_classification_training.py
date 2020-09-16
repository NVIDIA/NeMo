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

from typing import Any, Optional, List

from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

import pytorch_lightning as ptl
from omegaconf import MISSING, DictConfig, OmegaConf

from nemo.utils import logging
from nemo.collections.cv.models import ResNet50

from nemo.core.config import hydra_runner, DataLoaderConfig, TrainerConfig, AdamConfig
from nemo.collections.cv.datasets.configs import CIFAR10Config
from nemo.collections.cv.modules import ImageEncoderConfig

import PIL


@dataclass
class AppConfig:
    """
    This is structured config for this application.
    Args:
        trainer: configuration of the trainer.
        model: configuation of the model.
    """

    dataloader: DataLoaderConfig = DataLoaderConfig()
    dataset: CIFAR10Config = CIFAR10Config()
    transforms: Optional[Any] = None
    model: ImageEncoderConfig = ImageEncoderConfig()
    optim: AdamConfig = AdamConfig()
    trainer: TrainerConfig = TrainerConfig()


# Register schema.
cs = ConfigStore.instance()
cs.store(node=AppConfig, name="cifar10_resnet50_image_classification_training")

# Load configuration file from "conf" dir using schema for validation/retrieving the default values.
@hydra_runner(config_path="conf", config_name="cifar10_resnet50_image_classification_training")
def main(cfg: AppConfig):
    # Show configuration.
    logging.info("Application settings\n" + OmegaConf.to_yaml(cfg))

    # Instantiate the "model".
    resnet = ResNet50(cfg.model)

    # Instantiate the dataloader/dataset.
    train_dl = resnet.instantiate_dataloader(cfg.dataloader, cfg.dataset, cfg.transforms)

    # Setup the optimization.
    resnet.setup_optimization(cfg.optim)

    # Create the trainer.
    trainer = ptl.Trainer(**(cfg.trainer))

    # Train the model on dataset.
    trainer.fit(model=resnet, train_dataloader=train_dl)


if __name__ == "__main__":
    main()
