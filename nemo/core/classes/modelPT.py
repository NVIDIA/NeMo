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

import copy
import inspect
import tarfile
import tempfile
from abc import abstractmethod
from os import path
from typing import Dict, Optional, Union

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, Trainer
from torch import load, save

from nemo.core import optim
from nemo.core.classes.common import Model, Serialization
from nemo.core.optim import prepare_lr_scheduler
from nemo.utils import logging

__all__ = ['ModelPT']

_MODEL_CONFIG_YAML = "model_config.yaml"
_MODEL_WEIGHTS = "model_weights.ckpt"


class ModelPT(LightningModule, Model):
    """
    Interface for Pytorch-lightning based NeMo models
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """
        Base class from which all NeMo models should inherit
        Args:
            cfg (DictConfig):  configuration object.
                The cfg object should have (optionally) the following sub-configs:
                    * train_ds - to instantiate training dataset
                    * validation_ds - to instantiate validation dataset
                    * test_ds - to instantiate testing dataset
                    * optim - to instantiate optimizer with learning rate scheduler

            trainer (Optional): Pytorch Lightning Trainer instance
        """
        if not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg constructor argument must be of type DictConfig but got {type(cfg)} instead.")
        if trainer is not None and not isinstance(trainer, Trainer):
            raise ValueError(
                f"trainer constructor argument must be either None or pytroch_lightning.Trainer. But got {type(trainer)} instead."
            )
        super().__init__()
        self._cfg = cfg
        self.save_hyperparameters(self._cfg)
        self._train_dl = None
        self._validation_dl = None
        self._test_dl = None
        self._optimizer = None
        self._scheduler = None
        self._trainer = trainer

        if self._cfg is not None:
            if 'train_ds' in self._cfg and self._cfg.train_ds is not None:
                self.setup_training_data(self._cfg.train_ds)
            if 'validation_ds' in self._cfg and self._cfg.validation_ds is not None:
                self.setup_validation_data(self._cfg.validation_ds)
            if 'test_ds' in self._cfg and self._cfg.test_ds is not None:
                self.setup_test_data(self._cfg.test_ds)

    def save_to(self, save_path: str):
        """
        Saves model instance (weights and configuration) into .nemo file. You can use "restore_from" method to fully
        restore instance from .nemo file.

        .nemo file is an archive (tar.gz) with the following:
            model_config.yaml - model configuration in .yaml format. You can deserialize this into cfg argument for model's constructor
            model_wights.chpt - model checkpoint

        Args:
            save_path: Path to .nemo file where model instance should be saved
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Prepare filenames.
            config_yaml_file = path.join(tmpdir, _MODEL_CONFIG_YAML)
            model_weights_file = path.join(tmpdir, _MODEL_WEIGHTS)
            # Save the configuration.
            self.to_config_file(path2yaml_file=config_yaml_file)
            # Save the weights.
            save(self.state_dict(), model_weights_file)
            # Put both to .nemo file.
            self.__make_nemo_file_from_folder(filename=save_path, source_dir=tmpdir)

    @classmethod
    def restore_from(cls, restore_path: str):
        """
        Restores model instance (weights and configuration) into .nemo file
        Args:
            restore_path: path to .nemo file from which model should be instantiated

            Example:
                ```
                model = nemo.collections.asr.models.EncDecCTCModel.restore_from('asr.nemo')
                assert isinstance(model, nemo.collections.asr.models.EncDecCTCModel)
                ```

        Returns:
            An instance of type cls
        """
        if not path.exists(restore_path):
            raise FileExistsError(f"Can't find {restore_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            cls.__unpack_nemo_file(path2file=restore_path, out_folder=tmpdir)
            # Prepare filenames.
            config_yaml_file = path.join(tmpdir, _MODEL_CONFIG_YAML)
            model_weights_file = path.join(tmpdir, _MODEL_WEIGHTS)

            # Load the configuration.
            config = OmegaConf.load(config_yaml_file)
            print(config)

            # Instantiate the object.
            instance = Serialization.from_config_dict(config)

            # Load the weights.
            state_dict = load(model_weights_file, map_location=lambda storage, loc: storage)
            instance.load_state_dict(state_dict)

        return instance

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
        # If config was not explicitly passed to us
        if optim_config is None:
            # See if internal config has `optim` namespace
            if self._cfg is not None and hasattr(self._cfg, 'optim'):
                optim_config = self._cfg.optim

        # If config is still None, or internal config has no Optim, return without instantiation
        if optim_config is None:
            logging.info('No optimizer config provided, therefore no optimizer was created')
            return

        # Setup optimizer and scheduler
        if optim_config is not None and isinstance(optim_config, DictConfig):
            optim_config = OmegaConf.to_container(optim_config)

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
                optim_config['sched']['iters_per_batch'] = iters_per_batch
            else:
                optim_config['sched']['max_steps'] = self._trainer.max_steps

        # Force into DictConfig from nested structure
        optim_config = OmegaConf.create(optim_config)
        # Get back nested dict so we its mutable
        optim_config = OmegaConf.to_container(optim_config, resolve=True)

        # Extract scheduler config if inside optimizer config
        if 'sched' in optim_config:
            scheduler_config = optim_config.pop('sched')
        else:
            scheduler_config = None

        # Check if caller provided optimizer name, default to Adam otherwise
        optimizer_cls = optim_config.get('cls', None)

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
        if 'args' in optim_config:
            optimizer_args = optim_config.pop('args')
            optimizer_args = optim.parse_optimizer_args(optimizer_name, optimizer_args)
        else:
            optimizer_args = copy.deepcopy(optim_config)

            # Remove extra parameters from optimizer_args nest
            # Assume all other parameters are to be passed into optimizer constructor
            optimizer_args.pop('name', None)
            optimizer_args.pop('cls', None)
            optimizer_args.pop('lr', None)

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
            optimizer=self._optimizer, scheduler_config=scheduler_config, train_dataloader=self._train_dl
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

    @property
    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def __make_nemo_file_from_folder(filename, source_dir):
        with tarfile.open(filename, "w:gz") as tar:
            # tar.add(source_dir, arcname=path.basename(source_dir))
            tar.add(source_dir, arcname="./")

    @staticmethod
    def __unpack_nemo_file(path2file: str, out_folder: str) -> str:
        if not path.exists(path2file):
            raise FileNotFoundError(f"{path2file} does not exist")
        tar = tarfile.open(path2file, "r:gz")
        tar.extractall(path=out_folder)
        tar.close()
        return out_folder
