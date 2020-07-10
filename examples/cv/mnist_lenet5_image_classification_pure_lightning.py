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

from nemo.collections.cv.models import MNISTLeNet5, MNISTLeNet5Config
from nemo.core.config import Config, TrainerConfig, set_config
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

    name: str = "Training of a LeNet-5 Model using a pure PyTorchLightning approach - using DDP on 2 GPUs."
    trainer: TrainerConfig = TrainerConfig(gpus=2, distributed_backend="ddp")
    model: MNISTLeNet5Config = MNISTLeNet5Config()


@set_config(config=AppConfig)
def main(cfg: DictConfig):
    # Show configuration - user can modify every parameter from command line!
    logging.info("=" * 40 + " Hydra says hello! " + "=" * 40 + "\n" + cfg.pretty())

    # The "model" - with dataloader/dataset inside of it.
    lenet5 = MNISTLeNet5(cfg.model)

    # Setup train data loader and optimizer
    lenet5.setup_training_data()

    # Setup optimizer and scheduler
    if 'sched' in cfg.model.optim:
        if cfg.trainer.max_steps is None:
            if cfg.trainer.gpus == 0:
                # training on CPU
                iters_per_batch = cfg.trainer.max_epochs / float(
                    cfg.trainer.num_nodes * cfg.trainer.accumulate_grad_batches
                )
            else:
                iters_per_batch = cfg.trainer.max_epochs / float(
                    cfg.trainer.gpus * cfg.trainer.num_nodes * cfg.trainer.accumulate_grad_batches
                )
            cfg.model.optim.sched.iters_per_batch = iters_per_batch
        else:
            cfg.model.optim.sched.max_steps = cfg.trainer.max_steps

    # Setup optimizer and scheduler
    lenet5.setup_optimization()

    # Create trainer.
    trainer = ptl.Trainer(**(cfg.trainer))

    # Train.
    trainer.fit(model=lenet5)


if __name__ == "__main__":
    main()
