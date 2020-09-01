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
from omegaconf import OmegaConf

import pytorch_lightning as ptl
from omegaconf import DictConfig, MISSING

from nemo.collections.cv.models import MNISTLeNet5
from nemo.utils import logging
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

import hydra


@hydra_runner(config_path="conf", config_name="mnist_lenet5_image_classification")
def main(cfg: DictConfig):
    # Show configuration - user can modify every parameter from command line!
    logging.info("Application config\n" + OmegaConf.to_yaml(cfg))

    # Create trainer.
    trainer = ptl.Trainer(**(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    # The "model" - with dataloader/dataset inside of it.
    lenet5 = MNISTLeNet5(cfg.model, trainer=trainer)

    # Train.
    trainer.fit(model=lenet5)


if __name__ == "__main__":
    main()
