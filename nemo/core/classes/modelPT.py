# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from dataclasses import is_dataclass
from os import path
from typing import Callable, Dict, List, Optional, Union

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only

from nemo.core import optim
from nemo.core.classes.common import Model
from nemo.core.optim import prepare_lr_scheduler
from nemo.utils import logging, model_utils
from nemo.utils.app_state import AppState
from nemo.utils.get_rank import is_global_rank_zero

# Need to set them before EFF import as it is using them.
_MODEL_CONFIG_YAML = "model_config.yaml"
_MODEL_WEIGHTS = "model_weights.ckpt"

__all__ = ['ModelPT']

"""
Internal global flags that determine core functionality of ModelPT.

_MODEL_IS_RESTORED:
    This flag determines the context of the model - whether the model is currently being
    restored or not.
    -   When set, it can be assumed that the model's will disable all automatic methods -
        setup_training_data(), setup_validation/test_data() and their multi equivalents.
    -   If a model is being restored from a archive file (tarfile), it can be assumed that
        under this context, the cwd is *inside* the tarfile itself.

_MODEL_RESTORE_PATH:
    A string path to a a file from which the model is being restored.
    This file can either be a PyTorch Lightning Checkpoint, or a archive (tarfile) that contains
    artifact objects.
    If it is an archive file, during restoration, the cwd will be temporarily moved to inside the
    archive itself.
"""
_MODEL_IS_RESTORED = False
_MODEL_RESTORE_PATH = None


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
        if trainer is not None and not isinstance(trainer, Trainer):
            raise ValueError(
                f"trainer constructor argument must be either None or pytroch_lightning.Trainer. But got {type(trainer)} instead."
            )
        super().__init__()

        # Convert config to a DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)

        # Convert config to support Hydra 1.0+ instantiation
        cfg = model_utils.maybe_update_config_version(cfg)

        if 'target' not in cfg:
            # This is for Jarvis service.
            OmegaConf.set_struct(cfg, False)
            cfg.target = "{0}.{1}".format(self.__class__.__module__, self.__class__.__name__)
            OmegaConf.set_struct(cfg, True)

        self._cfg = cfg

        self.save_hyperparameters(self._cfg)
        self._train_dl = None
        self._validation_dl = None
        self._test_dl = None
        self._optimizer = None
        self._scheduler = None
        self._trainer = trainer

        # Set device_id in AppState
        if torch.cuda.is_available() and torch.cuda.current_device() is not None:
            app_state = AppState()
            app_state.device_id = torch.cuda.current_device()

        if self._cfg is not None and not self._is_model_being_restored():
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

        # ModelPT wrappers over subclass implementations
        self.training_step = model_utils.wrap_training_step(self.training_step)

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
            self.artifacts = {}
        if self.artifacts is None:
            self.artifacts = {}
        if src is not None and src.strip() != '':
            archive_item = model_utils.ArtifactItem()

            basename_src = os.path.basename(src)
            # filename exists in current workdir - use it and raise warning
            # this case is during model restoration or when file is written to cwd.
            if os.path.exists(basename_src):
                logging.warning(f"Using {os.path.abspath(basename_src)} instead of {src}.")
                used_src = basename_src

                # Case: register_artifact() called inside restoration context
                if self._is_model_being_restored() and self._is_restore_type_tarfile():
                    archive_item.path_type = model_utils.ArtifactPathType.TAR_PATH
                else:
                    archive_item.path_type = model_utils.ArtifactPathType.LOCAL_PATH

            else:
                used_src = src
                archive_item.path_type = model_utils.ArtifactPathType.LOCAL_PATH

            if not os.path.exists(used_src):
                # File not found in local path or by basename
                # Try to locate it inside the .nemo archive (if model was restored)
                # Case: register_artifact() called outside restoration context
                if self._is_restore_type_tarfile():
                    # Get path where the command is executed - the artifacts will be "retrieved" there
                    # (original .nemo behavior)
                    cwd = os.getcwd()
                    try:
                        # Step into the nemo archive to try and find the file
                        with tempfile.TemporaryDirectory() as tmpdir:
                            self.__unpack_nemo_file(path2file=_MODEL_RESTORE_PATH, out_folder=tmpdir)
                            os.chdir(tmpdir)
                            if os.path.exists(basename_src):
                                logging.warning(f"Using {os.path.abspath(basename_src)} instead of {src}.")
                                used_src = basename_src

                                archive_item.path = used_src
                                archive_item.path_type = model_utils.ArtifactPathType.TAR_PATH
                            else:
                                # No further action can be taken, file not found anywhere
                                raise FileNotFoundError(
                                    f"Could not find {used_src} inside "
                                    f"tarfile {_MODEL_RESTORE_PATH} or under local"
                                )
                    finally:
                        # change back working directory
                        os.chdir(cwd)
                else:
                    # No further action can be taken, file not found anywhere
                    raise FileNotFoundError(f"Could not find {used_src}")
            else:
                # Found filepath
                archive_item.path = used_src

            # But disregarding whether you use "local" or "remote" artifact - always store the original path.
            # This fixes issues raising when finetuning NLP models that create and register tokenizer vocabs.
            if config_path in self.artifacts:
                logging.warning(
                    f"Artifact {config_path} with value '{self.artifacts[config_path]}' "
                    f"already exists and will be overwritten with value '{src}'!"
                )

            self.artifacts[config_path] = archive_item
            return used_src
        else:
            return src

    def _default_save_to(self, save_path: str):
        """
        Saves model instance (weights and configuration) into .nemo file.
        You can use "restore_from" method to fully restore instance from .nemo file.

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
                for (conf_path, src) in self.artifacts.items():  # type: (str, model_utils.ArtifactItem)
                    try:
                        if src.path_type == model_utils.ArtifactPathType.LOCAL_PATH and os.path.exists(src.path):
                            shutil.copy2(src.path, tmpdir)
                        elif src.path_type == model_utils.ArtifactPathType.TAR_PATH:
                            # Need to step into nemo archive to extract file
                            # Get path where the command is executed - the artifacts will be "retrieved" there
                            # (original .nemo behavior)
                            cwd = os.getcwd()
                            try:
                                # Step into the nemo archive to try and find the file
                                with tempfile.TemporaryDirectory() as archive_dir:
                                    self.__unpack_nemo_file(path2file=_MODEL_RESTORE_PATH, out_folder=archive_dir)
                                    os.chdir(archive_dir)
                                    shutil.copy2(src.path, tmpdir)
                            finally:
                                # change back working directory
                                os.chdir(cwd)
                        else:
                            raise ValueError(f"Invalid ArchivePathType found: {src.path_type}")
                    except Exception:
                        logging.error(f"Could not copy artifact {src} used in {conf_path}")

            self.to_config_file(path2yaml_file=config_yaml)
            torch.save(self.state_dict(), model_weights)
            self.__make_nemo_file_from_folder(filename=save_path, source_dir=tmpdir)

    @rank_zero_only
    def save_to(self, save_path: str):
        """
        Saves model instance (weights and configuration) into .nemo file
         You can use "restore_from" method to fully restore instance from .nemo file.

        .nemo file is an archive (tar.gz) with the following:
            model_config.yaml - model configuration in .yaml format. You can deserialize this into cfg argument for model's constructor
            model_wights.chpt - model checkpoint

        Args:
            save_path: Path to .nemo file where model instance should be saved
        """

        # Add NeMo rank check as well
        if not is_global_rank_zero():
            return
        else:
            self._default_save_to(save_path)

    @classmethod
    def _default_restore_from(
        cls,
        restore_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = False,
        return_config: bool = False,
    ):
        """
        Restores model instance (weights and configuration) into .nemo file
        Args:
            restore_path: path to .nemo file from which model should be instantiated
            override_config_path: path to a yaml config that will override the internal
                config file or an OmegaConf / DictConfig object representing the model config.
            map_location: Optional torch.device() to map the instantiated model to a device.
                By default (None), it will select a GPU if available, falling back to CPU otherwise.
            strict: Passed to load_state_dict.
            return_config: If set to true, will return just the underlying config of the restored
                model as an OmegaConf DictConfig object without instantiating the model.

            Example:
                ```
                model = nemo.collections.asr.models.EncDecCTCModel.restore_from('asr.nemo')
                assert isinstance(model, nemo.collections.asr.models.EncDecCTCModel)
                ```

        Returns:
            An instance of type cls or its underlying config (if return_config is set).
        """
        # Get path where the command is executed - the artifacts will be "retrieved" there
        # (original .nemo behavior)
        cwd = os.getcwd()

        if map_location is None:
            if torch.cuda.is_available():
                map_location = torch.device('cuda')
            else:
                map_location = torch.device('cpu')

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                cls._set_model_restore_state(is_being_restored=True)
                cls.__unpack_nemo_file(path2file=restore_path, out_folder=tmpdir)
                os.chdir(tmpdir)
                if override_config_path is None:
                    config_yaml = path.join(tmpdir, _MODEL_CONFIG_YAML)
                else:
                    # can be str path or OmegaConf / DictConfig object
                    config_yaml = override_config_path
                if not isinstance(config_yaml, (OmegaConf, DictConfig)):
                    conf = OmegaConf.load(config_yaml)
                else:
                    conf = config_yaml
                if override_config_path is not None:
                    # Resolve the override config
                    conf = OmegaConf.to_container(conf, resolve=True)
                    conf = OmegaConf.create(conf)
                    # If override is top level config, extract just `model` from it
                    if 'model' in conf:
                        conf = conf.model

                if return_config:
                    instance = conf
                else:
                    model_weights = path.join(tmpdir, _MODEL_WEIGHTS)
                    OmegaConf.set_struct(conf, True)
                    instance = cls.from_config_dict(config=conf)
                    instance = instance.to(map_location)
                    instance.load_state_dict(torch.load(model_weights, map_location=map_location), strict=strict)

                    logging.info(f'Model {cls.__name__} was successfully restored from {restore_path}.')
            finally:
                cls._set_model_restore_state(is_being_restored=False)
                os.chdir(cwd)

        return instance

    @classmethod
    def restore_from(
        cls,
        restore_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = False,
        return_config: bool = False,
    ):
        """
        Restores model instance (weights and configuration) from .nemo file.

        Args:
            restore_path: path to .nemo file from which model should be instantiated
            override_config_path: path to a yaml config that will override the internal
                config file or an OmegaConf / DictConfig object representing the model config.
            map_location: Optional torch.device() to map the instantiated model to a device.
                By default (None), it will select a GPU if available, falling back to CPU otherwise.
            strict: Passed to load_state_dict.
            return_config: If set to true, will return just the underlying config of the restored
                model as an OmegaConf DictConfig object without instantiating the model.

            Example:
                ```
                model = nemo.collections.asr.models.EncDecCTCModel.restore_from('asr.nemo')
                assert isinstance(model, nemo.collections.asr.models.EncDecCTCModel)
                ```

        Returns:
            An instance of type cls or its underlying config (if return_config is set).
        """
        if not path.exists(restore_path):
            raise FileNotFoundError(f"Can't find {restore_path}")

        global _MODEL_RESTORE_PATH
        _MODEL_RESTORE_PATH = os.path.abspath(os.path.expanduser(restore_path))
        return cls._default_restore_from(restore_path, override_config_path, map_location, strict, return_config)

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
            cls._set_model_restore_state(is_being_restored=True)

            checkpoint = super().load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                *args,
                map_location=map_location,
                hparams_file=hparams_file,
                strict=strict,
                **kwargs,
            )

        finally:
            cls._set_model_restore_state(is_being_restored=False)
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
        Setups data loader to be used in validation
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
        self._val_dl_idx = 0
        self._validation_names = None
        self._validation_dl = None  # type: torch.utils.data.DataLoader

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        try:
            self._multi_dataset_mode = True
            model_utils.resolve_validation_dataloaders(model=self)
        finally:
            self._multi_dataset_mode = False

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
        self._test_dl_idx = 0
        self._test_names = None
        self._test_dl = None  # type: torch.utils.data.DataLoader

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        try:
            self._multi_dataset_mode = True
            model_utils.resolve_test_dataloaders(model=self)
        finally:
            self._multi_dataset_mode = False

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
                if self._cfg.optim is None:
                    self._cfg.optim = copy.deepcopy(optim_config)
                else:
                    with open_dict(self._cfg.optim):
                        self._cfg.optim = copy.deepcopy(optim_config)

        # Setup optimizer and scheduler
        if optim_config is not None and isinstance(optim_config, DictConfig):
            optim_config = OmegaConf.to_container(optim_config, resolve=True)

        if self._trainer is None:
            logging.warning(f"Trainer wasn't specified in model constructor. Make sure that you really wanted it.")

        if 'sched' in optim_config and self._trainer is not None:
            if not isinstance(self._trainer.accumulate_grad_batches, int):
                raise ValueError("We do not currently support gradient acculumation that is not an integer.")
            if self._trainer.max_steps is None:
                # Store information needed to calculate max_steps
                optim_config['sched']['t_max_epochs'] = self._trainer.max_epochs
                optim_config['sched']['t_accumulate_grad_batches'] = self._trainer.accumulate_grad_batches
                optim_config['sched']['t_limit_train_batches'] = self._trainer.limit_train_batches
                if self._trainer.distributed_backend is None:
                    optim_config['sched']['t_num_workers'] = self._trainer.num_gpus or 1
                elif self._trainer.distributed_backend == "ddp_cpu":
                    optim_config['sched']['t_num_workers'] = self._trainer.num_processes * self._trainer.num_nodes
                elif self._trainer.distributed_backend == "ddp":
                    optim_config['sched']['t_num_workers'] = self._trainer.num_gpus * self._trainer.num_nodes
                else:
                    logging.warning(
                        f"The lightning trainer received accelerator: {self._trainer.distributed_backend}. We "
                        "recommend to use 'ddp' instead."
                    )
                    optim_config['sched']['t_num_workers'] = self._trainer.num_gpus * self._trainer.num_nodes
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
        optimizer_cls = optim_config.get('_target_', None)

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

        # Adaptive schedulers don't need `lr`
        if lr is not None:
            optimizer_args['lr'] = lr

        # Actually instantiate the optimizer
        if optimizer_cls is not None:
            if inspect.isclass(optimizer_cls):
                optimizer = optimizer_cls(self.parameters(), **optimizer_args)
                logging.info("Optimizer config = %s", str(optimizer))

                self._optimizer = optimizer

            else:
                # Attempt class path resolution
                try:
                    optimizer_cls = OmegaConf.create({'_target_': optimizer_cls})
                    if lr is not None:
                        optimizer_config = {'lr': lr}
                    else:
                        optimizer_config = {}
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
            optimizer = optimizer(self.parameters(), **optimizer_args)

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

    def val_dataloader(self):
        if self._validation_dl is not None:
            return self._validation_dl

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    def validation_epoch_end(
        self, outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Default DataLoader for Validation set which automatically supports multiple data loaders
        via `multi_validation_epoch_end`.

        If multi dataset support is not required, override this method entirely in base class.
        In such a case, there is no need to implement `multi_validation_epoch_end` either.

        .. note::
            If more than one data loader exists, and they all provide `val_loss`,
            only the `val_loss` of the first data loader will be used by default.
            This default can be changed by passing the special key `val_dl_idx: int`
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
            output_dict = self.multi_validation_epoch_end(outputs, dataloader_idx=0)

            if output_dict is not None and 'log' in output_dict:
                self.log_dict(output_dict.pop('log'), on_epoch=True)

            return output_dict

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
                    if 'val_loss' not in output_dict and dataloader_idx == self._val_dl_idx:
                        output_dict['val_loss'] = dataloader_logs['val_loss']

                # For every item in the result dictionary
                for k, v in dataloader_logs.items():
                    # If the key is `log`
                    if k == 'log':
                        # Parse every element of the log, and attach the prefix name of the data loader
                        log_dict = {}

                        for k_log, v_log in v.items():
                            # If we are logging the metric, but dont provide it at result level,
                            # store it twice - once in log and once in result level.
                            # Also mark log with prefix name to avoid log level clash with other data loaders
                            if k_log not in output_dict['log'] and dataloader_idx == self._val_dl_idx:
                                new_k_log = k_log

                                # Also insert duplicate key with prefix for ease of comparison / avoid name clash
                                log_dict[dataloader_prefix + k_log] = v_log

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

            if 'log' in output_dict:
                self.log_dict(output_dict.pop('log'), on_epoch=True)

            # return everything else
            return output_dict

    def test_epoch_end(
        self, outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Default DataLoader for Test set which automatically supports multiple data loaders
        via `multi_test_epoch_end`.

        If multi dataset support is not required, override this method entirely in base class.
        In such a case, there is no need to implement `multi_test_epoch_end` either.

        .. note::
            If more than one data loader exists, and they all provide `test_loss`,
            only the `test_loss` of the first data loader will be used by default.
            This default can be changed by passing the special key `test_dl_idx: int`
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
            output_dict = self.multi_test_epoch_end(outputs, dataloader_idx=0)

            if output_dict is not None and 'log' in output_dict:
                self.log_dict(output_dict.pop('log'), on_epoch=True)

            return output_dict

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
                    if 'test_loss' not in output_dict and dataloader_idx == self._test_dl_idx:
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
                            if k_log not in output_dict['log'] and dataloader_idx == self._test_dl_idx:
                                new_k_log = k_log

                                # Also insert duplicate key with prefix for ease of comparison / avoid name clash
                                log_dict[dataloader_prefix + k_log] = v_log

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

            if 'log' in output_dict:
                self.log_dict(output_dict.pop('log'), on_epoch=True)

            # return everything else
            return output_dict

    def multi_validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]], dataloader_idx: int = 0
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Adds support for multiple validation datasets. Should be overriden by subclass,
        so as to obtain appropriate logs for each of the dataloaders.

        Args:
            outputs: Same as that provided by LightningModule.validation_epoch_end()
                for a single dataloader.
            dataloader_idx: int representing the index of the dataloader.

        Returns:
            A dictionary of values, optionally containing a sub-dict `log`,
            such that the values in the log will be pre-pended by the dataloader prefix.
        """
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
        """
        Adds support for multiple test datasets. Should be overriden by subclass,
        so as to obtain appropriate logs for each of the dataloaders.

        Args:
            outputs: Same as that provided by LightningModule.validation_epoch_end()
                for a single dataloader.
            dataloader_idx: int representing the index of the dataloader.

        Returns:
            A dictionary of values, optionally containing a sub-dict `log`,
            such that the values in the log will be pre-pended by the dataloader prefix.
        """
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

    def teardown(self, stage: str):
        """
        Called at the end of fit and test.

        Args:
            stage: either 'fit' or 'test'
        """
        if stage == 'fit':
            # Update env variable to bypass multi gpu issue after training
            # This fix affects usage of trainer.test() after trainer.train()
            # If trainer.train() was done on multiple GPUs, then trainer.test()
            # will try to do ddp, even if its a new Trainer object with just 1 GPU.
            # Temporary patch to fix that
            if 'PL_TRAINER_GPUS' in os.environ:
                os.environ.pop('PL_TRAINER_GPUS')

        super().teardown(stage)

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
            ::
            state_dict = nemo.collections.asr.models.EncDecCTCModel.extract_state_dict_from('asr.nemo', './asr_ckpts')


            To restore a model from a Model level checkpoint
            ::
            model = nemo.collections.asr.models.EncDecCTCModel(cfg)  # or any other method of restoration
            model.load_state_dict(torch.load("./asr_ckpts/model_weights.ckpt"))


            To convert the .nemo tarfile into multiple Module level PyTorch checkpoints
            ::
            state_dict = nemo.collections.asr.models.EncDecCTCModel.extract_state_dict_from('asr.nemo', './asr_ckpts', split_by_module=True)


            To restore a module from a Module level checkpoint
            ::
            model = nemo.collections.asr.models.EncDecCTCModel(cfg)  # or any other method of restoration

            # load the individual components
            model.preprocessor.load_state_dict(torch.load("./asr_ckpts/preprocessor.ckpt"))
            model.encoder.load_state_dict(torch.load("./asr_ckpts/encoder.ckpt"))
            model.decoder.load_state_dict(torch.load("./asr_ckpts/decoder.ckpt"))


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

    def prepare_test(self, trainer: 'Trainer') -> bool:
        """
        Helper method to check whether the model can safely be tested
        on a dataset after training (or loading a checkpoint).

        ::

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

    def set_trainer(self, trainer: Trainer):
        """
        Set an instance of Trainer object.

        Args:
            trainer: PyTorch Lightning Trainer object.
        """
        self._trainer = trainer
        self.set_world_size(self._trainer)

    def set_world_size(self, trainer: Trainer):
        """
        Determines the world size from the PyTorch Lightning Trainer.
        And then updates AppState.

        Args:
            trainer (Trainer): PyTorch Lightning Trainer object
        """
        # Update AppState with world information from trainer
        if isinstance(trainer, Trainer):
            app_state = AppState()
            if self._trainer.num_gpus and self._trainer.num_nodes:
                app_state.world_size = self._trainer.num_gpus * self._trainer.num_nodes
        else:
            logging.warning(f'World size can only be set by PyTorch Lightning Trainer.')

    def _update_dataset_config(self, dataset_name: str, config: Optional[Union[DictConfig, Dict]]):
        """
        Update the config (if not None) of the dataset by given name.
        Preserves said config after updating.

        Args:
            dataset_name: str name of the dataset whose config is being updated.
                Can be one of `train`, `validation` and `test`.
            config: Optional DictConfig or dict. If None is passed, this method simply returns.
                If dict is passed, it is cast into a DictConfig.
                The internal config is updated with the passed config.
        """
        if hasattr(self, '_multi_dataset_mode') and self._multi_dataset_mode is True:
            return

        if config is not None:
            if not isinstance(config, DictConfig):
                config = OmegaConf.create(config)

            if dataset_name in ['train', 'validation', 'test']:
                OmegaConf.set_struct(self.cfg, False)

                key_name = dataset_name + "_ds"
                self.cfg[key_name] = config

                OmegaConf.set_struct(self.cfg, True)

                # Update hyper parameters by calling property setter
                self.cfg = self._cfg
            else:
                raise ValueError("`dataset_name` when updating config must be one of [train, validation, test]")

    @property
    def num_weights(self):
        """
        Utility property that returns the total number of parameters of the Model.
        """
        num: int = 0
        for p in self.parameters():
            if p.requires_grad:
                num += p.numel()
        return num

    @property
    def cfg(self):
        """
        Property that holds the finalized internal config of the model.

        Note:
            Changes to this config are not reflected in the state of the model.
            Please create a new model using an updated config to properly update the model.
        """
        return self._cfg

    @cfg.setter
    def cfg(self, cfg):
        """
        Property that holds the finalized internal config of the model.

        Note:
            Changes to this config are not reflected in the state of the model.
            Please create a new model using an updated config to properly update the model.
        """
        self._cfg = cfg
        self._set_hparams(self._cfg)
        self._hparams_initial = copy.deepcopy(self._hparams)

    @staticmethod
    def __make_nemo_file_from_folder(filename, source_dir):
        with tarfile.open(filename, "w:gz") as tar:
            # tar.add(source_dir, arcname=path.basename(source_dir))
            tar.add(source_dir, arcname=".")

    @staticmethod
    def __unpack_nemo_file(path2file: str, out_folder: str) -> str:
        if not path.exists(path2file):
            raise FileNotFoundError(f"{path2file} does not exist")
        tar = tarfile.open(path2file, "r:gz")
        tar.extractall(path=out_folder)
        tar.close()
        return out_folder

    @staticmethod
    def _is_model_being_restored() -> bool:
        global _MODEL_IS_RESTORED
        return _MODEL_IS_RESTORED

    @staticmethod
    def _set_model_restore_state(is_being_restored: bool):
        global _MODEL_IS_RESTORED
        _MODEL_IS_RESTORED = is_being_restored

    @staticmethod
    def _is_restore_type_tarfile() -> bool:
        """
        Utility method that checks if the restore path of the underlying Model
        is a tarfile (can be any valid archive)._MODEL_EFF_SAVE
        """
        global _MODEL_RESTORE_PATH

        if _MODEL_RESTORE_PATH is None:
            return False
        else:
            if tarfile.is_tarfile(_MODEL_RESTORE_PATH):
                return True
            else:
                return False

    @staticmethod
    def set_eff_save(use_eff_save: bool):
        global _MODEL_EFF_SAVE
        _MODEL_EFF_SAVE = use_eff_save

    @staticmethod
    def use_eff_save() -> bool:
        global _MODEL_EFF_SAVE
        return _MODEL_EFF_SAVE
