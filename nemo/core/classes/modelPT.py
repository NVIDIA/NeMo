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

import inspect
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule
from torch.optim.optimizer import Optimizer

from nemo.core import optim
from nemo.core.classes.common import Model
from nemo.core.config.base_config import Config
from nemo.core.optim import prepare_lr_scheduler
from nemo.utils import logging

__all__ = ['ModelPT', 'ModelPTConfig']


@dataclass
class ModelPTConfig(Config):
    """Inherit from this class when you parametrize NeMo models"""

    optim: Any = None
    train_ds: Any = None
    validation_ds: Any = None
    test_ds: Any = None
    trainer: Any = None


class ModelPT(LightningModule, Model):
    """
    Interface for Pytorch-lightning based NeMo models
    """

    def __init__(self, cfg: ModelPTConfig = None, trainer=None):
        super().__init__()
        self._cfg = cfg
        self.save_hyperparameters(self._cfg)
        self._train_dl = None
        self._validation_dl = None
        self._test_dl = None
        self._optimizer = None
        self._scheduler = None
        self._trainer = trainer

        if cfg is not None:
            if 'train_ds' in cfg and cfg.train_ds is not None:
                self.setup_training_data(cfg.train_ds)
            if 'validation_ds' in cfg and cfg.validation_ds is not None:
                self.setup_validation_data(cfg.validation_ds)
            if 'test_ds' in cfg and cfg.test_ds is not None:
                self.setup_test_data(cfg.test_ds)

    @abstractmethod
    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        """
        Setups data loader to be used in training
        Args:
            train_data_layer_config: training data layer parameters.
        Returns:

        """
        pass

    @abstractmethod
    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        """
        (Optionally) Setups data loader to be used in validation
        Args:
            val_data_layer_config: validation data layer parameters.
        Returns:

        """
        pass

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        """
        (Optionally) Setups data loader to be used in test
        Args:
            test_data_layer_config: test data layer parameters.
        Returns:

        """
        raise NotImplementedError()

    def setup_optimization(self, optim_config: Optional[Union[DictConfig, Dict]] = None):
        """
        Prepares an optimizer from a string name and its optional config parameters.

        Args:
            optim_config: a dictionary containing the following keys.
                - "lr": mandatory key for learning rate. Will raise ValueError
                if not provided.

                - "optimizer": string name pointing to one of the available
                optimizers in the registry. If not provided, defaults to "adam".

                - "opt_args": Optional list of strings, in the format "arg_name=arg_value".
                The list of "arg_value" will be parsed and a dictionary of optimizer
                kwargs will be built and supplied to instantiate the optimizer.
        """
        # Setup optimizer and scheduler
        if 'sched' in optim_config and self._trainer is not None:
            if not isinstance(self._trainer.accumulate_grad_batches, int):
                raise ValueError("We do not currently support gradient acculumation that is not an integer.")
            if self._trainer.max_steps is None:
                if self._trainer.num_gpus == 0:
                    # training on CPU
                    iters_per_batch = self._trainer.max_epochs / float(
                        self._trainer.num_nodes * self._trainer.accumulate_grad_batches
                    )
                else:
                    iters_per_batch = self._trainer.max_epochs / float(
                        self._trainer.num_gpus * self._trainer.num_nodes * self._trainer.accumulate_grad_batches
                    )
                optim_config.sched.iters_per_batch = iters_per_batch
            else:
                optim_config.sched.max_steps = self._trainer.max_steps

        optim_config = optim_config or {}  # In case null was passed as optim_params

        # Force into DictConfig from nested structure
        optim_config = OmegaConf.create(optim_config)

        # Check if caller provided optimizer name, default to Adam otherwise
        optimizer_cls = optim_config.get('cls', None)

        logging.info(f"CLS : {optimizer_cls}")

        if optimizer_cls is None:
            # Try to get optimizer name for dynamic resolution, defaulting to Adam
            optimizer_name = optim_config.get('name', 'adam')
        else:
            if inspect.isclass(optimizer_cls):
                optimizer_name = optimizer_cls.__name__.lower()
            else:
                # resolve the class name (lowercase) from the class path if not provided
                optimizer_name = optimizer_cls.split(".")[-1].lower()

        # We are guarenteed to have lr since it is required by the argparser
        # But maybe user forgot to pass it to this function
        lr = optim_config.get('lr', None)

        if 'lr' is None:
            raise ValueError('`lr` must be passed to `optimizer_config` when setting up the optimization !')

        # Check if caller has optimizer kwargs, default to empty dictionary
        optimizer_args = optim_config.get('args', {})
        optimizer_args = optim.parse_optimizer_args(optimizer_name, optimizer_args)

        # Actually instantiate the optimizer
        if optimizer_cls is not None:
            if inspect.isclass(optimizer_cls):
                optimizer = optimizer_cls(self.parameters(), lr=lr, **optimizer_args)
                logging.info("Optimizer config = %s", str(optimizer))

                self._optimizer = optimizer

            else:
                # Attempt class path resolution
                try:
                    optimizer_cls = OmegaConf.create({'cls': optimizer_cls})
                    optimizer_config = {'lr': lr}
                    optimizer_config.update(optimizer_args)

                    logging.info("About to instantiate optimizer")

                    optimizer_instance = hydra.utils.instantiate(
                        optimizer_cls, self.parameters(), **optimizer_config
                    )  # type: DictConfig

                    logging.info("Optimizer config = %s", str(optimizer_instance))

                    self._optimizer = optimizer_instance

                except Exception as e:
                    logging.error(
                        "Could not instantiate class path - {} with kwargs {}".format(
                            optimizer_cls, str(optimizer_config)
                        )
                    )
                    raise e

        else:
            optimizer = optim.get_optimizer(optimizer_name)
            optimizer = optimizer(self.parameters(), lr=lr, **optimizer_args)

            logging.info("Optimizer config = %s", str(optimizer))

            self._optimizer = optimizer

        self._scheduler = prepare_lr_scheduler(
            optimizer=self._optimizer, scheduler_config=optim_config, train_dataloader=self._train_dl
        )

    def configure_optimizers(self):
        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]

    def train_dataloader(self):
        if self._train_dl is not None:
            return self._train_dl

    def val_dataloader(self):
        if self._validation_dl is not None:
            return self._validation_dl

    # def test_dataloader(self):
    #     if self._test_dl is not None:
    #         return self._test_dl
