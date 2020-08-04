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

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LoggerCollection as _LoggerCollection
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.logging import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from nemo.constants import NEMO_ENV_VARNAME_DATETIME
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.lightning_logger_patch import add_filehandlers_to_pl_logger
from nemo.utils.exceptions import NeMoBaseException


class NotFoundError(NeMoBaseException):
    """ Raised when a file or folder is not found"""


class LoggerMisconfigurationError(NeMoBaseException):
    """ Raised when a mismatch between trainer.logger and exp_manager occurs"""

    def __init__(self, message):
        message = (
            message
            + " You can disable lighning's trainer from creating a logger by passing logger=False to its constructor."
        )
        super().__init__(message)


class CheckpointMisconfigurationError(NeMoBaseException):
    """ Raised when a mismatch between trainer.callbacks and exp_manager occurs"""


@dataclass
class ExpManagerConfig:
    # Log dir creation parameters
    explicit_log_dir: Optional[str] = None
    root_dir: Optional[str] = None
    name: Optional[str] = None
    version: Optional[str] = None
    use_datetime_version: Optional[bool] = True
    resume: Optional[bool] = False
    previous_log_dir: Optional[str] = None
    # Logging parameters
    create_tensorboard_logger: Optional[bool] = True
    summary_writter_kwargs: Optional[Dict] = None
    create_wandb_logger: Optional[bool] = False
    wandb_logger_kwargs: Optional[Dict] = None
    # Checkpointing parameters
    create_checkpoint_callback: Optional[bool] = True
    # Additional exp_manager arguments
    files_to_copy: Optional[List[str]] = None


def exp_manager(trainer: 'pytorch_lightning.Trainer', cfg: Optional[Union[DictConfig, Dict]] = None) -> Path:
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

    returns:
        directory (Path): The final logging directory where logging files are saved. Usually the concatenation of
            root_dir, name, and version.
    """
    if cfg is None:
        logging.error("exp_manager did not receive a cfg argument. It will be disabled.")
        return

    # Ensure passed cfg is compliant with ExpManagerConfig
    schema = OmegaConf.structured(ExpManagerConfig)
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    elif not isinstance(cfg, DictConfig):
        raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
    cfg = OmegaConf.merge(schema, cfg)

    error_checks(trainer, cfg)  # Ensures that trainer options are compliant with NeMo and exp_manager arguments

    log_dir, root_dir, name, version = get_log_dir(
        trainer=trainer,
        root_dir=cfg.root_dir,
        name=cfg.name,
        version=cfg.version,
        explicit_log_dir=cfg.explicit_log_dir,
        use_datetime_version=cfg.use_datetime_version,
        resume=cfg.resume,
        previous_log_dir=cfg.previous_log_dir,
    )
    cfg.name = name
    cfg.version = version

    # Create the logging directory if it does not exist
    os.makedirs(log_dir, exist_ok=True)  # Cannot limit creation to global zero as all ranks write to own log file
    logging.info(f'Experiments will be logged at {log_dir}')
    trainer.default_root_dir = log_dir

    # Handle Loggers by creating file and handle DEBUG statements
    # Note: trainer.global_rank and trainer.is_global_zero are not set until trainer.fit, so have to hack around it
    global_rank = trainer.node_rank * trainer.num_gpus + trainer.local_rank
    log_file = log_dir / f'nemo_log_globalrank-{global_rank}_localrank-{trainer.local_rank}.txt'
    logging.add_file_handler(log_file)
    logging.rank = global_rank

    if is_global_rank_zero():
        if cfg.create_tensorboard_logger or cfg.create_wandb_logger:
            configure_loggers(
                trainer,
                root_dir,
                cfg.name,
                cfg.version,
                cfg.create_tensorboard_logger,
                cfg.summary_writter_kwargs,
                cfg.create_wandb_logger,
                cfg.wandb_logger_kwargs,
            )

        if cfg.create_checkpoint_callback:
            configure_checkpointing(trainer, log_dir, cfg.name)

        # Move files_to_copy to folder and add git information if present
        if cfg.files_to_copy:
            for _file in cfg.files_to_copy:
                copyfile(Path(_file), log_dir)

        # Create files for cmd args and git info
        with open(log_dir / 'cmd-args.log', 'w') as _file:
            _file.write(" ".join(sys.argv))

        # Try to get git hash
        git_repo, git_hash = get_git_hash()
        if git_repo:
            with open(log_dir / 'git-info.log', 'w') as _file:
                _file.write(f'commit hash: {git_hash}')
                _file.write(get_git_diff())

        # Add err_file logging to global_rank zero
        logging.add_err_file_handler(log_dir / 'nemo_error_log.txt')

        # Add lightning file logging to global_rank zero
        add_filehandlers_to_pl_logger(log_dir / 'lightning_logs.txt', log_dir / 'nemo_error_log.txt')

    return log_dir


def error_checks(trainer: 'pytorch_lightning.Trainer', cfg: Optional[Union[DictConfig, Dict]] = None):
    """
    Checks that the passed trainer is compliant with NeMo and exp_manager's passed configuration. Checks that:
        - Throws error when hydra has changed the working directory. This causes issues with lightning's DDP
        - Throws error when trainer has loggers defined but create_tensorboard_logger or create_WandB_logger is True
        - Prints error messages when 1) run on multi-node and not slurm, and 2) run on multi-gpu without DDP
    """
    if HydraConfig.initialized() and get_original_cwd() != os.getcwd():
        raise ValueError(
            "Hydra changed the working directory. This interferes with ExpManger's functionality. Please pass "
            "hydra.run.dir=. to your python script."
        )
    if trainer.logger is not None and (cfg.create_tensorboard_logger or cfg.create_wandb_logger):
        raise LoggerMisconfigurationError(
            "The pytorch lightning trainer that was passed to exp_manager contained a logger, and either "
            f"create_tensorboard_logger: {cfg.create_tensorboard_logger} or create_wandb_logger: {cfg.create_wandb_logger} "
            "was set to True. These can only be used if trainer does not already have a logger."
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


def check_resume(trainer, previous_log_dir, explicit_log_dir, root_dir, name, version, resume_past_end=False):
    if trainer.logger is not None:
        raise LoggerMisconfigurationError(
            "The pytorch lightning trainer that was passed to exp_manager contained a logger and resume was set to "
            "True. Please remove the logger from the lightning trainer."
        )
    if explicit_log_dir or root_dir or name or version:
        logging.error(
            f"exp_manager received resume == True, and at least one of explicit_log_dir: {explicit_log_dir}, root_dir: "
            f"{root_dir}, name: {name}, or version: {version}. Please note that "
            "explicit_log_dir, root_dir, name, and version will be ignored."
        )

    checkpoint_dir = Path(Path(previous_log_dir) / "checkpoints")
    checkpoint = None
    if not checkpoint_dir.exists():
        raise NotFoundError(f"There was no checkpoint folder at previous_log_dir :{previous_log_dir}. Cannot resume.")
    elif checkpoint_dir.match("*end.ckpt"):
        if resume_past_end:
            if len(checkpoint_dir.glob('*end.ckpt')) > 1:
                raise ValueError(
                    f"Multiple multiple checkpoints {checkpoint_dir.glob('*end.ckpt')} that matches *end.ckpt."
                )
            logging.info(f"Resuming from {checkpoint_dir.glob('*end.ckpt')}")
            checkpoint = list(checkpoint_dir.glob('*end.ckpt'))[0]
        else:
            raise ValueError(
                f"Found {checkpoint_dir.glob('*end.ckpt')} indicating that the last training run has already "
                "completed."
            )
    elif not checkpoint_dir.match("*last.ckpt"):
        raise NotFoundError(f"There were no checkpoints found in {previous_log_dir}. Is the folder correct?")
    elif len(checkpoint_dir.glob("*last.ckpt")) > 1:
        raise ValueError(f"Multiple multiple checkpoints {checkpoint_dir.glob('*last.ckpt')} that matches *last.ckpt.")
    else:
        checkpoint = list(checkpoint_dir.glob('*last.ckpt'))[0]

    trainer.resume_from_checkpoint = checkpoint

    if is_global_rank_zero():
        # Move old files to a new folder
        other_run_dirs = checkpoint_dir.glob("run_*")
        run_count = 0
        for fold in other_run_dirs:
            if fold.is_dir():
                run_count += 1
        new_run_dir = Path(checkpoint_dir / f"run_{run_count}")
        new_run_dir.mkdir()
        for child in checkpoint_dir.iterdir():
            if child.is_file():
                copyfile(child, new_run_dir)
    return Path(previous_log_dir), previous_log_dir, "", ""


def check_explicit_log_dir(trainer, explicit_log_dir, root_dir, name, version):
    if trainer.logger is not None:
        raise LoggerMisconfigurationError(
            "The pytorch lightning trainer that was passed to exp_manager contained a logger and explicit_log_dir: "
            f"{explicit_log_dir} was pass to exp_manager. Please remove the logger from the lightning trainer."
        )
    if root_dir or name or version:
        logging.error(
            f"exp_manager received explicit_log_dir: {explicit_log_dir} and at least one of root_dir: {root_dir}, "
            f"name: {name}, or version: {version}. Please note that root_dir, "
            "name, and version will be ignored."
        )
    if is_global_rank_zero() and Path(explicit_log_dir).exists():
        logging.warning("Exp_manager is logging to {explicit_log_dir}, but it already exists.")
    return Path(explicit_log_dir), explicit_log_dir, "", ""


def get_log_dir(
    trainer: 'pytorch_lightning.Trainer',
    root_dir: str = None,
    name: str = None,
    version: str = None,
    explicit_log_dir: str = None,
    use_datetime_version: bool = True,
    resume: bool = False,
    previous_log_dir: str = None,
) -> Path:
    """
    Obtains the log_dir used for exp_manager.

    Returns:
        Path

    Raise:
        ValueError: If trainer is incompatible with arguments
    """
    if resume:  # If resuming from another checkpoint, short circuit
        return check_resume(trainer, previous_log_dir, explicit_log_dir, root_dir, name, version)

    if explicit_log_dir:  # If explicit log_dir was pass, short circuit
        return check_explicit_log_dir(trainer, explicit_log_dir, root_dir, name, version)

    # Default root_dir to ./NeMo_experiments if None was passed
    _root_dir = root_dir
    if root_dir is None:
        _root_dir = str(Path.cwd() / 'NeMo_experiments')

    # If the user has already defined a logger for the trainer, use the logger defaults for logging directory
    if trainer.logger is not None:
        if trainer.logger.save_dir:
            if root_dir:
                raise LoggerMisconfigurationError(
                    "The pytorch lightning trainer that was passed to exp_manager contained a logger, the logger's "
                    f"save_dir was not None, and root_dir ({root_dir}) was not None. If trainer.logger.save_dir "
                    "exists, exp_manager will use trainer.logger.save_dir as the logging directory and root_dir "
                    "must be None."
                )
            _root_dir = trainer.logger.save_dir
        if name:
            raise LoggerMisconfigurationError(
                "The pytorch lightning trainer that was passed to exp_manager contained a logger, and name: "
                f"{name} was also passed to exp_manager. If the trainer contains a "
                "logger, exp_manager will use trainer.logger.name, and name passed to exp_manager must be None."
            )
        name = trainer.logger.name
        version = trainer.logger.version
    # Use user-defined root_dir, project_name, exp_name, and versioning options
    else:
        version = None
        if use_datetime_version:
            version = os.environ.get(NEMO_ENV_VARNAME_DATETIME, None)
            if trainer.is_slurm_managing_tasks:
                logging.warning("Running on a slurm cluster. Versioning by datetime will not work.")
            elif is_global_rank_zero():
                version = time.strftime('%Y-%m-%d_%H-%M-%S')
                os.environ[NEMO_ENV_VARNAME_DATETIME] = version

        name = name or "default"
        log_dir_wo_version = Path(_root_dir) / name

        # Always create TensorBoardLogger, so we can retrieve version if running on slurm
        tensorboard_logger = TensorBoardLogger(save_dir=log_dir_wo_version, version=version)
        if version is None:
            version = tensorboard_logger.version

    log_dir = Path(_root_dir) / Path(str(name)) / Path(str(version))
    return log_dir, _root_dir, name, version


def get_git_hash():
    """
    Helper function that tries to get the commit hash if running inside a git folder

    returns:
        Bool: Whether the git subprocess ran without error
        str: git subprocess output or error message
    """
    try:
        return (
            True,
            subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode(),
        )
    except subprocess.CalledProcessError as err:
        return False, "{}\n".format(err.output.decode("utf-8"))


def get_git_diff():
    """
    Helper function that tries to get the git diff if running inside a git folder

    returns:
        Bool: Whether the git subprocess ran without error
        str: git subprocess output or error message
    """
    try:
        return subprocess.check_output(['git', 'diff'], stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as err:
        return "{}\n".format(err.output.decode("utf-8"))


class LoggerList(_LoggerCollection):
    def __init__(self, _logger_iterable, nemo_name=None, nemo_version=""):
        super().__init__(_logger_iterable)
        self._nemo_name = nemo_name
        self._nemo_version = nemo_version

    @property
    def name(self) -> str:
        return self._nemo_name

    @property
    def version(self) -> str:
        return self._nemo_version


def configure_loggers(
    trainer,
    root_dir,
    name,
    version,
    create_tensorboard_logger,
    summary_writter_kwargs,
    create_wandb_logger,
    wandb_kwargs,
):
    # Potentially create tensorboard logger and/or WandBLogger
    logger_list = []
    if create_tensorboard_logger:
        if summary_writter_kwargs is None:
            summary_writter_kwargs = {}
        elif "log_dir" in summary_writter_kwargs:
            raise ValueError(
                "You cannot pass `log_dir` as part of `summary_writter_kwargs`. `log_dir` is handled by lightning's "
                "TensorBoardLogger logger."
            )
        tensorboard_logger = TensorBoardLogger(save_dir=root_dir, name=name, version=version, **summary_writter_kwargs)
        logger_list.append(tensorboard_logger)
        logging.info("TensorboardLogger has been set up")

    if create_wandb_logger:
        if "name" not in wandb_kwargs and "project" not in wandb_kwargs:
            raise ValueError("name and project are required for wandb_logger")
        wandb_logger = WandbLogger(save_dir=root_dir, version=version, **wandb_kwargs)

        logger_list.append(wandb_logger)
        logging.info("WandBLogger has been set up")

    logger_list = (
        LoggerList(logger_list, nemo_name=name, nemo_version=version) if len(logger_list) > 1 else logger_list[0]
    )
    trainer.configure_logger(logger_list)


def configure_checkpointing(trainer, log_dir, name):
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            raise CheckpointMisconfigurationError(
                "The pytorch lightning trainer that was passed to exp_manager contained a ModelCheckpoint "
                "and create_checkpoint_callback was set to True. Please either set create_checkpoint_callback "
                "to False, or remove ModelCheckpoint from the lightning trainer"
            )
    if Path(trainer.weights_save_path) != Path.cwd():
        raise CheckpointMisconfigurationError(
            "The pytorch lightning was passed weights_save_path. This variable is ignored by exp_manager"
        )
    else:
        logging.warning("trainer had a weights_save_path of cwd(). This was ignored.")
    # Create the callback and attach it to trainer
    class NeMoModelCheckpoint(ModelCheckpoint):
        @rank_zero_only
        def on_train_end(self, trainer, pl_module):
            filepath = os.path.join(self.dirpath, self.prefix + 'end.ckpt')
            try:  # Try lightning master signature
                self._save_model(filepath, trainer, pl_module)  # noqa pylint: disable=too-many-function-args
            except TypeError:  # Fall back to lightning == 0.8.5 signature if failed
                self._save_model(filepath)  # noqa

    checkpoint_callback = NeMoModelCheckpoint(
        filepath=Path(log_dir / 'checkpoints' / '{val_loss:.2f}-{epoch}'),
        save_top_k=3,
        save_last=True,
        prefix=name + "--",
    )
    trainer.configure_checkpoint_callback(checkpoint_callback)
    trainer.callbacks.append(checkpoint_callback)
    trainer.checkpoint_callback = checkpoint_callback


def find_last_checkpoint(log_dir, resume_past_end=False):
    checkpoint_dir = Path(Path(log_dir) / "checkpoints")
    if not checkpoint_dir.exists():
        raise NotFoundError(f"There was no checkpoint folder at log_dir :{log_dir}. Cannot resume.")
    elif checkpoint_dir.match("*end.ckpt"):
        if resume_past_end:
            if len(checkpoint_dir.glob('*end.ckpt')) > 1:
                raise ValueError(
                    f"Multiple multiple checkpoints {checkpoint_dir.glob('*end.ckpt')} that matches *end.ckpt."
                )
            logging.info(f"Resuming from {checkpoint_dir.glob('*end.ckpt')}")
            return list(checkpoint_dir.glob('*end.ckpt'))[0]
        else:
            raise ValueError(
                f"Found {checkpoint_dir.glob('*end.ckpt')} indicating that the last training run has already "
                "completed."
            )
    elif not checkpoint_dir.match("*last.ckpt"):
        raise NotFoundError(f"There were no checkpoints found in {log_dir}. Is the folder correct?")

    if len(checkpoint_dir.glob("*last.ckpt")) > 1:
        raise ValueError(f"Multiple multiple checkpoints {checkpoint_dir.glob('*last.ckpt')} that matches *last.ckpt.")
    return list(checkpoint_dir.glob('*last.ckpt'))[0]
