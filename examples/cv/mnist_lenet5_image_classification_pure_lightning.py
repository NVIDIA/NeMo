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

import pytorch_lightning as ptl

from nemo.core.config import Config, set_config, TrainerConfig
from dataclasses import dataclass
from omegaconf import DictConfig

from nemo.collections.cv.models import MNISTLeNet5, MNISTLeNet5Config

from nemo.utils import logging

@dataclass
class AppConfig(Config):
    """
    This is structured config for this application.

    Args:
        name: Description of the application.
        trainer: configuration of the trainer.
        model: configuation of the model.
    """
    name: str="Training of a LeNet-5 Model using a pure PyTorchLightning approach - using DDP on 2 GPUs."
    trainer: TrainerConfig=TrainerConfig(gpus=2, distributed_backend="ddp")
    model: MNISTLeNet5Config=MNISTLeNet5Config()


@set_config(config=AppConfig)
def main(cfg: DictConfig):

    # The "model" - with dataloader/dataset inside of it.
    lenet5 = MNISTLeNet5(cfg.model)

    # Create trainer.
    trainer = ptl.Trainer(**(cfg.trainer))

    # Train.
    trainer.fit(model=lenet5)

if __name__ == "__main__":
    main()