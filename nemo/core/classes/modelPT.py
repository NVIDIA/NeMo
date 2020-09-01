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
import os
import shutil
import tarfile
import tempfile
from abc import abstractmethod
from os import path
from typing import Callable, Dict, List, Optional, Union

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, Trainer

from nemo.collections.common import callbacks
from nemo.core import optim
from nemo.core.classes.common import Model
from nemo.core.optim import prepare_lr_scheduler
from nemo.utils import logging, model_utils

__all__ = ['ModelPT']

_MODEL_CONFIG_YAML = "model_config.yaml"
_MODEL_WEIGHTS = "model_weights.ckpt"
_MODEL_IS_RESTORED = False


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
        if 'target' not in cfg:
            # This is for Jarvis service.
            OmegaConf.set_struct(cfg, False)
            cfg.target = "{0}.{1}".format(self.__class__.__module__, self.__class__.__name__)
            OmegaConf.set_struct(cfg, True)

        config = OmegaConf.to_container(cfg, resolve=True)
        config = OmegaConf.create(config)
        OmegaConf.set_struct(config, True)

        self._cfg = config

        self.save_hyperparameters(self._cfg)
        self._train_dl = None
        self._validation_dl = None
        self._test_dl = None
        self._optimizer = None
        self._scheduler = None
        self._trainer = trainer

        if self._cfg is not None and not self.__is_model_being_restored():
            if 'train_ds' in self._cfg and self._cfg.train_ds is not None:
                self.setup_training_data(self._cfg.train_ds)

            if 'validation_ds' in self._cfg and self._cfg.validation_ds is not None:
                self.setup_multiple_validation_data(val_data_config=None)

            if 'test_ds' in self._cfg and self._cfg.test_ds is not None:
                self.setup_multiple_test_data(test_data_config=None)

        else:
            if 'train_ds' in self._cfg and self._cfg.train_ds is not None:
                logging.warning(
                    f"Please call the ModelPT.setup_training_data() method "
                    f"and provide a valid configuration file to setup the train data loader.\n"
                    f"Train config : \n{OmegaConf.to_yaml(self._cfg.train_ds)}"
                )

            if 'validation_ds' in self._cfg and self._cfg.validation_ds is not None:
                logging.warning(
                    f"Please call the ModelPT.setup_validation_data() or ModelPT.setup_multiple_validation_data() method "
                    f"and provide a valid configuration file to setup the validation data loader(s). \n"
                    f"Validation config : \n{OmegaConf.to_yaml(self._cfg.validation_ds)}"
                )

            if 'test_ds' in self._cfg and self._cfg.test_ds is not None:
                logging.warning(
                    f"Please call the ModelPT.setup_test_data() or ModelPT.setup_multiple_test_data() method "
                    f"and provide a valid configuration file to setup the test data loader(s).\n"
                    f"Test config : \n{OmegaConf.to_yaml(self._cfg.test_ds)}"
                )

    def register_artifact(self, config_path: str, src: str):
        """
        Register model artifacts with this function. These artifacts (files) will be included inside .nemo file
        when model.save_to("mymodel.nemo") is called.

        WARNING: If you specified /example_folder/example.txt but ./example.txt exists, then ./example.txt will be used.

        Args:
            config_path: config path where artifact is used
            src: path to the artifact

        Returns:
            path to be used when accessing artifact. If src='' or None then '' or None will be returned
        """
        if not hasattr(self, 'artifacts'):
            self.artifacts = []
        if self.artifacts is None:
            self.artifacts = []
        if src is not None and src.strip() != '':
            basename_src = os.path.basename(src)
            # filename exists in current workdir - use it and raise warning
            if os.path.exists(basename_src):
                logging.warning(f"Using {os.path.abspath(basename_src)} instead of {src}.")
                used_src = basename_src
            else:
                used_src = src
            if not os.path.exists(used_src):
                raise FileNotFoundError(f"Could not find {used_src}")
            self.artifacts.append((config_path, used_src))
            return used_src
        else:
            return src

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
            config_yaml = path.join(tmpdir, _MODEL_CONFIG_YAML)
            model_weights = path.join(tmpdir, _MODEL_WEIGHTS)
            if hasattr(self, 'artifacts') and self.artifacts is not None:
                for (conf_path, src) in self.artifacts:
                    try:
                        if os.path.exists(src):
                            shutil.copy2(src, tmpdir)
                    except Exception:
                        logging.error(f"Could not copy artifact {src} used in {conf_path}")

            self.to_config_file(path2yaml_file=config_yaml)
            torch.save(self.state_dict(), model_weights)
            self.__make_nemo_file_from_folder(filename=save_path, source_dir=tmpdir)

    @classmethod
    def restore_from(cls, restore_path: str, override_config_path: Optional[str] = None):
        """
        Restores model instance (weights and configuration) into .nemo file
        Args:
            restore_path: path to .nemo file from which model should be instantiated
            override_config_path: path to a yaml config that will override the internal
                config file

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

        cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                cls.__set_model_restore_state(is_being_restored=True)
                cls.__unpack_nemo_file(path2file=restore_path, out_folder=tmpdir)
                os.chdir(tmpdir)
                if override_config_path is None:
                    config_yaml = path.join(tmpdir, _MODEL_CONFIG_YAML)
                else:
                    config_yaml = override_config_path
                conf = OmegaConf.load(config_yaml)
                if override_config_path is not None:
                    # Resolve the override config
                    conf = OmegaConf.to_container(conf, resolve=True)
                    conf = OmegaConf.create(conf)
                    # If override is top level config, extract just `model` from it
                    if 'model' in conf:
                        conf = conf.model
                model_weights = path.join(tmpdir, _MODEL_WEIGHTS)
                OmegaConf.set_struct(conf, True)
                instance = cls.from_config_dict(config=conf)
                instance.load_state_dict(torch.load(model_weights))

                logging.info(f'Model {cls.__name__} was successfully restored from {restore_path}.')
            finally:
                cls.__set_model_restore_state(is_being_restored=False)
                os.chdir(cwd)

        return instance

    @classmethod
    def extract_state_dict_from(cls, restore_path: str, save_dir: str, split_by_module: bool = False):
        """
        Extract the state dict(s) from a provided .nemo tarfile and save it to a directory.
        Args:
            restore_path: path to .nemo file from which state dict(s) should be extracted
            save_dir: directory in which the saved state dict(s) should be stored
            split_by_module: bool flag, which determins whether the output checkpoint should
                be for the entire Model, or the individual module's that comprise the Model

        Example:
            To convert the .nemo tarfile into a single Model level PyTorch checkpoint
            ```
            state_dict = nemo.collections.asr.models.EncDecCTCModel.extract_state_dict_from('asr.nemo', './asr_ckpts)
            ```

            To restore a model from a Model level checkpoint
            ```
            model = nemo.collections.asr.models.EncDecCTCModel(cfg)  # or any other method of restoration
            model.load_state_dict(torch.load("./asr_ckpts/model_weights.ckpt"))
            ```

            To convert the .nemo tarfile into multiple Module level PyTorch checkpoints
            ```
            state_dict = nemo.collections.asr.models.EncDecCTCModel.extract_state_dict_from('asr.nemo', './asr_ckpts,
                                                                                             split_by_module=True)
            ```

            To restore a module from a Module level checkpoint
            ```
            model = model = nemo.collections.asr.models.EncDecCTCModel(cfg)  # or any other method of restoration

            # load the individual components
            model.preprocessor.load_state_dict(torch.load("./asr_ckpts/preprocessor.ckpt"))
            model.encoder.load_state_dict(torch.load("./asr_ckpts/encoder.ckpt"))
            model.decoder.load_state_dict(torch.load("./asr_ckpts/decoder.ckpt"))
            ```

        Returns:
            The state dict that was loaded from the original .nemo checkpoint
        """
        if not path.exists(restore_path):
            raise FileExistsError(f"Can't find {restore_path}")

        cwd = os.getcwd()

        save_dir = os.path.abspath(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                cls.__unpack_nemo_file(path2file=restore_path, out_folder=tmpdir)
                os.chdir(tmpdir)
                model_weights = path.join(tmpdir, _MODEL_WEIGHTS)
                state_dict = torch.load(model_weights)

                if not split_by_module:
                    filepath = os.path.join(save_dir, _MODEL_WEIGHTS)
                    torch.save(state_dict, filepath)

                else:
                    key_set = set([key.split(".")[0] for key in state_dict.keys()])
                    for primary_key in key_set:
                        inner_keys = [key for key in state_dict.keys() if key.split(".")[0] == primary_key]
                        state_dict_subset = {
                            ".".join(inner_key.split(".")[1:]): state_dict[inner_key] for inner_key in inner_keys
                        }
                        filepath = os.path.join(save_dir, f"{primary_key}.ckpt")
                        torch.save(state_dict_subset, filepath)

                logging.info(f'Checkpoints from {restore_path} were successfully extracted into {save_dir}.')
            finally:
                os.chdir(cwd)

        return state_dict

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        *args,
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        **kwargs,
    ):
        """
        Loads ModelPT from checkpoint, with some maintenance of restoration.
        For documentation, please refer to LightningModule.load_from_checkpoin() documentation.
        """
        checkpoint = None
        try:
            cls.__set_model_restore_state(is_being_restored=True)

            checkpoint = super().load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                *args,
                map_location=map_location,
                hparams_file=hparams_file,
                strict=strict,
                **kwargs,
            )

        finally:
            cls.__set_model_restore_state(is_being_restored=False)

        return checkpoint

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

    def setup_multiple_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        """
        (Optionally) Setups data loader to be used in validation, with support for multiple data loaders.

        Args:
            val_data_layer_config: validation data layer parameters.
        """
        # Set some placeholder overriden by helper method
        self._validation_loss_idx = 0
        self._validation_names = None
        self._validation_dl = None  # type: torch.utils.data.DataLoader

        if val_data_config is not None:
            if isinstance(val_data_config, dict):
                val_data_config = DictConfig(val_data_config)

            self._cfg.validation_ds = val_data_config

        model_utils.resolve_validation_dataloaders(model=self)

        if self._validation_names is None:
            if self._validation_dl is not None and type(self._validation_dl) in [list, tuple]:
                self._validation_names = ['val_{}_'.format(idx) for idx in range(len(self._validation_dl))]

    def setup_multiple_test_data(self, test_data_config: Union[DictConfig, Dict]):
        """
        (Optionally) Setups data loader to be used in test, with support for multiple data loaders.

        Args:
            test_data_layer_config: test data layer parameters.
        """
        # Set some placeholder overriden by helper method
        self._test_loss_idx = 0
        self._test_names = None
        self._test_dl = None  # type: torch.utils.data.DataLoader

        if test_data_config is not None:
            if isinstance(test_data_config, dict):
                test_data_config = DictConfig(test_data_config)

            self._cfg.test_ds = test_data_config

        model_utils.resolve_test_dataloaders(model=self)

        if self._test_names is None:
            if self._test_dl is not None and type(self._test_dl) in [list, tuple]:
                self._test_names = ['test_{}_'.format(idx) for idx in range(len(self._test_dl))]

    def setup_optimization(self, optim_config: Optional[Union[DictConfig, Dict]] = None):
        """
        Prepares an optimizer from a string name and its optional config parameters.

        Args:
            optim_config: A dictionary containing the following keys:

                * "lr": mandatory key for learning rate. Will raise ValueError if not provided.
                * "optimizer": string name pointing to one of the available optimizers in the registry. \
                If not provided, defaults to "adam".
                * "opt_args": Optional list of strings, in the format "arg_name=arg_value". \
                The list of "arg_value" will be parsed and a dictionary of optimizer kwargs \
                will be built and supplied to instantiate the optimizer.
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

        else:
            # Preserve the configuration
            if not isinstance(optim_config, DictConfig):
                optim_config = OmegaConf.create(optim_config)

            # See if internal config has `optim` namespace before preservation
            if self._cfg is not None and hasattr(self._cfg, 'optim'):
                self._cfg.optim = optim_config

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

        # Try to instantiate scheduler for optimizer
        self._scheduler = prepare_lr_scheduler(
            optimizer=self._optimizer, scheduler_config=scheduler_config, train_dataloader=self._train_dl
        )

        # Return the optimizer with/without scheduler
        # This return allows multiple optimizers or schedulers to be created
        return self._optimizer, self._scheduler

    def configure_optimizers(self):
        self.setup_optimization()

        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]

    def train_dataloader(self):
        if self._train_dl is not None:
            return self._train_dl

    def training_step(self, batch, batch_ix):
        pass

    def val_dataloader(self):
        if self._validation_dl is not None:
            return self._validation_dl

    def validation_step(self, batch, batch_ix):
        return {}

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    def test_step(self, batch, batch_ix):
        return {}

    def validation_epoch_end(
        self, outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Default DataLoader for Validation set which automatically supports multiple data loaders
        via `multi_validation_epoch_end`.

        If multi dataset support is not required, override this method entirely in base class.
        In such a case, there is no need to implement `multi_validation_epoch_end` either.

        Note:
            If more than one data loader exists, and they all provide `val_loss`,
            only the `val_loss` of the first data loader will be used by default.
            This default can be changed by passing the special key `val_loss_idx: int`
            inside the `validation_ds` config.

        Args:
            outputs: Single or nested list of tensor outputs from one or more data loaders.

        Returns:
            A dictionary containing the union of all items from individual data_loaders,
            along with merged logs from all data loaders.
        """
        # Case where we dont provide data loaders
        if outputs is not None and len(outputs) == 0:
            return {}

        # Case where we provide exactly 1 data loader
        if type(outputs[0]) == dict:
            return self.multi_validation_epoch_end(outputs, dataloader_idx=0)

        else:  # Case where we provide more than 1 data loader
            output_dict = {'log': {}}

            # The output is a list of list of dicts, outer list corresponds to dataloader idx
            for dataloader_idx, val_outputs in enumerate(outputs):
                # Get prefix and dispatch call to multi epoch end
                dataloader_prefix = self.get_validation_dataloader_prefix(dataloader_idx)
                dataloader_logs = self.multi_validation_epoch_end(val_outputs, dataloader_idx=dataloader_idx)

                # If result was not provided, generate empty dict
                dataloader_logs = dataloader_logs or {}

                # Perform `val_loss` resolution first (if provided outside logs)
                if 'val_loss' in dataloader_logs:
                    if 'val_loss' not in output_dict and dataloader_idx == self._validation_loss_idx:
                        output_dict['val_loss'] = dataloader_logs['val_loss']

                # For every item in the result dictionary
                for k, v in dataloader_logs.items():
                    # If the key is `log`
                    if k == 'log':
                        # Parse every element of the log, and attach the prefix name of the data loader
                        log_dict = {}

                        for k_log, v_log in v.items():
                            # If we are logging the loss, but dont provide it at result level,
                            # store it twice - once in log and once in result level.
                            # Also mark log with prefix name to avoid log level clash with other data loaders
                            if (
                                k_log == 'val_loss'
                                and 'val_loss' not in output_dict['log']
                                and dataloader_idx == self._validation_loss_idx
                            ):
                                new_k_log = 'val_loss'

                                # Also insert duplicate key with prefix for ease of comparison / avoid name clash
                                log_dict[dataloader_prefix + k_log] = v_log

                            elif k_log == 'val_loss' and dataloader_idx != self._validation_loss_idx:
                                # replace all other "val_loss" with <prefix> + loss
                                # this avoid duplication of the word <prefix> + "val_loss" which causes confusion
                                new_k_log = dataloader_prefix + 'loss'

                            else:
                                # Simply prepend prefix to key and save
                                new_k_log = dataloader_prefix + k_log

                            # Store log value
                            log_dict[new_k_log] = v_log

                        # Update log storage of individual data loader
                        output_logs = output_dict['log']
                        output_logs.update(log_dict)

                        # Update global log storage
                        output_dict['log'] = output_logs

                    else:
                        # If any values are stored outside 'log', simply prefix name and store
                        new_k = dataloader_prefix + k
                        output_dict[new_k] = v

            return output_dict

    def test_epoch_end(
        self, outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Default DataLoader for Test set which automatically supports multiple data loaders
        via `multi_test_epoch_end`.

        If multi dataset support is not required, override this method entirely in base class.
        In such a case, there is no need to implement `multi_test_epoch_end` either.

        Note:
            If more than one data loader exists, and they all provide `test_loss`,
            only the `test_loss` of the first data loader will be used by default.
            This default can be changed by passing the special key `test_loss_idx: int`
            inside the `test_ds` config.

        Args:
            outputs: Single or nested list of tensor outputs from one or more data loaders.

        Returns:
            A dictionary containing the union of all items from individual data_loaders,
            along with merged logs from all data loaders.
        """
        # Case where we dont provide data loaders
        if outputs is not None and len(outputs) == 0:
            return {}

        # Case where we provide exactly 1 data loader
        if type(outputs[0]) == dict:
            return self.multi_test_epoch_end(outputs, dataloader_idx=0)

        else:  # Case where we provide more than 1 data loader
            output_dict = {'log': {}}

            # The output is a list of list of dicts, outer list corresponds to dataloader idx
            for dataloader_idx, test_outputs in enumerate(outputs):
                # Get prefix and dispatch call to multi epoch end
                dataloader_prefix = self.get_test_dataloader_prefix(dataloader_idx)
                dataloader_logs = self.multi_test_epoch_end(test_outputs, dataloader_idx=dataloader_idx)

                # If result was not provided, generate empty dict
                dataloader_logs = dataloader_logs or {}

                # Perform `test_loss` resolution first (if provided outside logs)
                if 'test_loss' in dataloader_logs:
                    if 'test_loss' not in output_dict and dataloader_idx == self._test_loss_idx:
                        output_dict['test_loss'] = dataloader_logs['test_loss']

                # For every item in the result dictionary
                for k, v in dataloader_logs.items():
                    # If the key is `log`
                    if k == 'log':
                        # Parse every element of the log, and attach the prefix name of the data loader
                        log_dict = {}
                        for k_log, v_log in v.items():
                            # If we are logging the loss, but dont provide it at result level,
                            # store it twice - once in log and once in result level.
                            # Also mark log with prefix name to avoid log level clash with other data loaders
                            if (
                                k_log == 'test_loss'
                                and 'test_loss' not in output_dict['log']
                                and dataloader_idx == self._test_loss_idx
                            ):
                                new_k_log = 'test_loss'

                                # Also insert duplicate key with prefix for ease of comparison
                                log_dict[dataloader_prefix + k_log] = v_log

                            elif k_log == 'test_loss' and dataloader_idx != self._test_loss_idx:
                                # replace all other "test_loss" with <prefix> + loss
                                # this avoid duplication of the word <prefix> + "test_loss" which causes confusion
                                new_k_log = dataloader_prefix + 'loss'

                            else:
                                # Simply prepend prefix to key and save
                                new_k_log = dataloader_prefix + k_log

                            log_dict[new_k_log] = v_log

                        # Update log storage of individual data loader
                        output_logs = output_dict.get('log', {})
                        output_logs.update(log_dict)

                        # Update global log storage
                        output_dict['log'] = output_logs

                    else:
                        # If any values are stored outside 'log', simply prefix name and store
                        new_k = dataloader_prefix + k
                        output_dict[new_k] = v

            return output_dict

    def multi_validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]], dataloader_idx: int = 0
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        logging.warning(
            "Multi data loader support has been enabled, but "
            "`multi_validation_epoch_end(outputs, dataloader_idx) has not been implemented.\n"
            "If you require multi data loader support for validation sets, please override this method.\n"
            "If you do not require multi data loader support, please instead override "
            "`validation_epoch_end(outputs)."
        )

    def multi_test_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]], dataloader_idx: int = 0
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        logging.warning(
            "Multi data loader support has been enabled, but "
            "`multi_test_epoch_end(outputs, dataloader_idx) has not been implemented.\n"
            "If you require multi data loader support for validation sets, please override this method.\n"
            "If you do not require multi data loader support, please instead override "
            "`test_epoch_end(outputs)."
        )

    def get_validation_dataloader_prefix(self, dataloader_idx: int = 0) -> str:
        """
        Get the name of one or more data loaders, which will be prepended to all logs.

        Args:
            dataloader_idx: Index of the data loader.

        Returns:
            str name of the data loader at index provided.
        """
        return self._validation_names[dataloader_idx]

    def get_test_dataloader_prefix(self, dataloader_idx: int = 0) -> str:
        """
        Get the name of one or more data loaders, which will be prepended to all logs.

        Args:
            dataloader_idx: Index of the data loader.

        Returns:
            str name of the data loader at index provided.
        """
        return self._test_names[dataloader_idx]

    def prepare_test(self, trainer: 'Trainer') -> bool:
        """
        Helper method to check whether the model can safely be tested
        on a dataset after training (or loading a checkpoint).

        # Usage:
        trainer = Trainer()
        if model.prepare_test(trainer):
            trainer.test(model)

        Returns:
            bool which declares the model safe to test. Provides warnings if it has to
            return False to guide the user.
        """
        if not hasattr(self._cfg, 'test_ds'):
            logging.info("No `test_ds` config found within the manifest.")
            return False

        # Replace ddp multi-gpu until PTL has a fix
        DDP_WARN = """\n\nDuring testing, it is currently advisable to construct a new Trainer "
                    "with single GPU and no DDP to obtain accurate results.
                    "Following pattern should be used: "
                    "gpu = 1 if cfg.trainer.gpus != 0 else 0"
                    "trainer = Trainer(gpus=gpu)"
                    "if model.prepare_test(trainer):"
                    "  trainer.test(model)\n\n"""

        if trainer is not None:
            if trainer.num_gpus > 1:
                logging.warning(DDP_WARN)
                return False

        # Assign trainer to the model
        self.set_trainer(trainer)
        return True

    def set_trainer(self, trainer: 'Trainer'):
        """
        Set an instance of Trainer object.

        Args:
            trainer: PyTorch Lightning Trainer object.
        """
        self._trainer = trainer

    @property
    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def cfg(self):
        return self._cfg

    @cfg.setter
    def cfg(self, cfg):
        self._cfg = cfg
        self._set_hparams(cfg)

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

    @staticmethod
    def __is_model_being_restored() -> bool:
        global _MODEL_IS_RESTORED
        return _MODEL_IS_RESTORED

    @staticmethod
    def __set_model_restore_state(is_being_restored: bool):
        global _MODEL_IS_RESTORED
        _MODEL_IS_RESTORED = is_being_restored
