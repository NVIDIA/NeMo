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
from typing import Any, Optional

import pytorch_lightning as ptl
from omegaconf import OmegaConf

from nemo.collections.cv.datasets.configs import MNISTConfig
from nemo.collections.cv.models import LeNet5
from nemo.core.config import AdamConfig, DataLoaderConfig, TrainerConfig, hydra_runner
from nemo.utils import logging


@dataclass
class AppConfig:
    """
    This is structured config for this application.
    Args:
        trainer: configuration of the trainer.
        model: configuation of the model.
    """

    dataloader: DataLoaderConfig = DataLoaderConfig()
    dataset: MNISTConfig = MNISTConfig()
    transforms: Optional[Any] = None  # List[Any] = field(default_factory=list) ?
    optim: AdamConfig = AdamConfig()
    trainer: TrainerConfig = TrainerConfig()


# Load configuration file from "conf" dir using schema for validation/retrieving the default values.
@hydra_runner(config_path="conf", config_name="mnist_lenet5_image_classification_training", schema=AppConfig)
def main(cfg: AppConfig):
    # Show configuration.
    logging.info("Application settings\n" + OmegaConf.to_yaml(cfg))

    # Instantiate the "model".
    lenet5 = LeNet5()

    # Instantiate the dataloader/dataset.
    train_dl = lenet5.instantiate_dataloader(cfg.dataloader, cfg.dataset, cfg.transforms)

    # Setup the optimization.
    lenet5.setup_optimization(cfg.optim)

    # Create the trainer.
    trainer = ptl.Trainer(**(cfg.trainer))

    # Train the model on dataset.
    trainer.fit(model=lenet5, train_dataloader=train_dl)


if __name__ == "__main__":
    main()
