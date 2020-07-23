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

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import Dict, List, Optional, Union

from hydra.utils import get_original_cwd
from omegaconf import MISSING, DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LoggerCollection as _LoggerCollection
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.logging import WandbLogger

from nemo.constants import NEMO_ENV_VARNAME_DATETIME
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero


@dataclass
class ExpManagerConfig:
    name: str = MISSING
    root_dir: Optional[str] = None
    create_tensorboard_logger: Optional[bool] = True
    create_checkpoint_callback: Optional[bool] = True
    files_to_copy: Optional[List[str]] = None
    wandb_exp: Optional[str] = None
    wandb_project: Optional[str] = None


def exp_manager(trainer: 'pytorch_lightning.Trainer', cfg: Optional[Union[DictConfig, Dict]] = None):
    """
    exp_manager is a helper function used to manage folders for experiments. It follows the pytorch lightning paradigm
    of root_dir/model_or_experiment_name/version. If the lightning trainer has a logger, exp_manager will get root_dir,
    name, and version from the logger. Otherwise it will use the root_dir and name arguments to create the logging
    directory. The version will be a datetime string if running single node, and version will be an integer if running
    on slurm multi-node.
    It optionally creates TensorBoardLogger, and ModelCheckpoint objects from pytorch lightning. It copies sys.argv,
    and git information if available to the logging directory. It creates a log file for each process to log their
    output into.

    Args:
        trainer (pytorch_lightning.Trainer): The lightning trainer.
        cfg (DictConfig, dict): Can have the following keys:
            - name (str): The name of the experiment. Required argument.
            - root_dir (str, Path): The base directory to create the logging directory. Defaults to None, which logs to
                ./NeMo_experiments.
            - create_tensorboard_logger (bool): Whether to create a tensorboard logger and attach it to the pytorch
                lightning trainer. Defaults to True.
            - create_checkpoint_callback (bool): Whether to create a ModelCheckpoint callback and attach it to the
                pytorch lightning trainer. The ModelCheckpoint saves the top 3 models with the best "val_loss" as well
                 as the most recent model. Defaults to True.
            - files_to_copy (list): A list of files to copy to the experiment logging directory. Defaults to None which
                copies no files.

            Following are optional values that can be provided to exp_manager inside cfg

            WandB support - Both arguments must be provided to optionally add WandB logging.
            - wandb_exp (str): The name of the WandB experiment. Must be provided if set as an argument.
            - wandb_porject (str): The project name of the WandB experiments. Groups together multiple experiments in
                the WandB dashboard. Must be provided if set as an argument.
    """
    if cfg is None:
        logging.error("exp_manager did not receive a cfg argument. It will be disabled.")
        return
    schema = OmegaConf.structured(ExpManagerConfig)
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    elif not isinstance(cfg, DictConfig):
        raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
    # Ensure passed cfg is compliant with ExpManagerConfig
    cfg = OmegaConf.merge(schema, cfg)

    if get_original_cwd() != os.getcwd():
        raise ValueError(
            "Hydra changed the working directory. This interferes with ExpManger's functionality. Please pass "
            "hydra.run.dir=. to your python script."
        )
    if trainer.num_nodes > 1 and not trainer.is_slurm_managing_tasks:
        logging.error(
            "You are running multi-node without slurm. Please note that this is not tested in NeMo and could result in "
            "errors."
        )
    if trainer.num_gpus > 1 and not trainer.use_ddp:
        logging.error(
            "You are running multi-gpu without ddp.Please note that this is not tested in NeMo and could result in "
            "errors."
        )
    # Default root_dir to ./NeMo_experiments if None was passed
    _root_dir = cfg.root_dir
    if cfg.root_dir is None:
        _root_dir = str(Path.cwd() / 'NeMo_experiments')
    # If the user has already defined a logger for the trainer, use the logger defaults for logging directory
    if trainer.logger is not None:
        if trainer.logger.save_dir:
            if cfg.root_dir:
                raise ValueError(
                    "The pytorch lightning trainer that was passed to exp_manager contained a logger, the logger's "
                    f"save_dir was not None, and root_dir ({cfg.root_dir}) was not None. If trainer.logger.save_dir "
                    "exists, exp_manager will use trainer.logger.save_dir as the logging directory and root_dir "
                    "must be None."
                )
            _root_dir = trainer.logger.save_dir
        if cfg.name:
            raise ValueError(
                "The pytorch lightning trainer that was passed to exp_manager contained a logger, and name "
                f"({cfg.name}) was also passed to exp_manager. If the trainer contains a logger, exp_manager will use "
                "trainer.logger.name, and name that is passed to exp_manager must be None."
            )
        if cfg.create_tensorboard_logger:
            raise ValueError(
                "The pytorch lightning trainer that was passed to exp_manager contained a logger, and "
                "create_tensorboard_logger was set to True. create_tensorboard_logger can only be used if trainer "
                "does not already have a logger."
            )
        name = trainer.logger.name
        version = trainer.logger.version
    else:
        version = os.environ.get(NEMO_ENV_VARNAME_DATETIME, None)
        if trainer.is_slurm_managing_tasks:
            logging.warning("Running on a slurm cluster. Versioning by datetime will not work.")
        elif is_global_rank_zero():
            version = time.strftime('%Y-%m-%d_%H-%M-%S')
            os.environ[NEMO_ENV_VARNAME_DATETIME] = version

        name = cfg.name
        if name is None:
            name = "default"

    # Always create TensorBoardLogger, so we can retrieve version if running on slurm
    tensorboard_logger = TensorBoardLogger(save_dir=_root_dir, name=name, version=version)
    if cfg.create_tensorboard_logger:
        # Attach logger to trainer
        trainer.configure_logger(tensorboard_logger)
    if version is None:
        version = tensorboard_logger.version

    # Potentially create a WandBLogger, if its arguments are provided
    if hasattr(cfg, 'wandb_exp') and hasattr(cfg, 'wandb_project'):
        if cfg.wandb_exp is not None and cfg.wandb_project is not None:
            wandb_logger = WandbLogger(
                name=cfg.wandb_exp, project=cfg.wandb_project, save_dir=_root_dir, version=version
            )

            logger_list = [tensorboard_logger, wandb_logger]
            logger_list = LoggerList(logger_list)

            trainer.configure_logger(logger_list)

            logging.info("WandBLogger has been setup in addition to TensorboardLogger")

    # Create the logging directory if it does not exist
    log_dir = Path(_root_dir, name, version)
    os.makedirs(log_dir, exist_ok=True)  # Cannot limit creation to global zero as all ranks write to own log file
    logging.info(f'Experiments will be logged at {log_dir}')
    if cfg.create_checkpoint_callback:
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                raise ValueError(
                    "The pytorch lightning trainer that was passed to exp_manager contained a ModelCheckpoint "
                    "and create_checkpoint_callback was set to True. Please either set create_checkpoint_callback "
                    "to False, or remove ModelCheckpoint from the lightning trainer"
                )
        if trainer.default_root_dir != trainer.weights_save_path:
            raise ValueError(
                "The pytorch lightning was passed weights_save_path. This variable is ignored by exp_manager"
            )
        trainer.weights_save_path = _root_dir
        trainer.default_root_dir = _root_dir
        # Create the callback and attach it to trainer
        checkpoint_callback = ModelCheckpoint(save_top_k=3, save_last=True, prefix=name + "--")
        trainer.configure_checkpoint_callback(checkpoint_callback)
        trainer.callbacks.append(checkpoint_callback)
        trainer.checkpoint_callback = checkpoint_callback

    # Move files_to_copy to folder and add git information if present
    if is_global_rank_zero():
        if cfg.files_to_copy:
            for _file in cfg.files_to_copy:
                basename = os.path.basename(_file)
                basename, ending = os.path.splitext(basename)
                basename = basename + f"_{version}" + ending
                copyfile(Path(_file), log_dir / basename)

        # Create files for cmd args and git info
        with open(log_dir / 'cmd-args.log', 'w') as _file:
            _file.write(" ".join(sys.argv))

        # Try to get git hash
        git_repo, git_hash = get_git_hash()
        if git_repo:
            with open(log_dir / 'git-info.log', 'w') as _file:
                _file.write(f'commit hash: {git_hash}')
                _file.write(get_git_diff())

    # Handle Loggers by creating file and handle DEBUG statements
    # Note: trainer.global_rank and trainer.is_global_zero are not set until trainer.fit, so have to hack around it
    global_rank = trainer.node_rank * trainer.num_processes + trainer.local_rank
    log_file = log_dir / f'log_globalrank-{global_rank}_localrank-{trainer.local_rank}.txt'
    logging.add_file_handler(log_file)
    logging.rank = global_rank


def get_git_hash():
    try:
        return (
            True,
            subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode(),
        )
    except subprocess.CalledProcessError as err:
        return False, "{}\n".format(err.output.decode("utf-8"))


def get_git_diff():
    try:
        return subprocess.check_output(['git', 'diff'], stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as err:
        return "{}\n".format(err.output.decode("utf-8"))


class LoggerList(_LoggerCollection):
    @property
    def name(self) -> str:
        logger_names = [str(logger.name) for logger in self._logger_iterable]
        return logger_names[0]

    @property
    def version(self) -> str:
        logger_versions = [str(logger.version) for logger in self._logger_iterable]
        return logger_versions[0]
