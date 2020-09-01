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
from shutil import copy, move
from typing import Any, Dict, List, Optional, Union

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LoggerCollection as _LoggerCollection
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from nemo.constants import NEMO_ENV_VARNAME_VERSION
from nemo.utils import logging
from nemo.utils.exceptions import NeMoBaseException
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.lightning_logger_patch import add_filehandlers_to_pl_logger


class CallbackManager:
    def __init__(self) -> None:
        self.callbacks = set(['LogEpochTimeCallback()', 'LogTrainValidLossCallback()'])

    def get_callback(self, callback_name: str):
        if callback_name in self.callbacks:
            return eval(callback_name)
        else:
            raise NameError("Provided Callback name is not part of nemo Callback system")

    def add_callback(self, callback_names: Union[str, List]):
        if type(callback_names) is str:
            callback_names = callback_names.split(',')

        callbacks = []
        for name in callback_names:
            callbacks.append(self.get_callback(name))

        return callbacks



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
    exp_dir: Optional[str] = None
    name: Optional[str] = None
    version: Optional[str] = None
    use_datetime_version: Optional[bool] = True
    resume_if_exists: Optional[bool] = False
    resume_past_end: Optional[bool] = False
    resume_ignore_no_checkpoint: Optional[bool] = False
    # Logging parameters
    create_tensorboard_logger: Optional[bool] = True
    summary_writer_kwargs: Optional[Dict[Any, Any]] = None
    create_wandb_logger: Optional[bool] = False
    wandb_logger_kwargs: Optional[Dict[Any, Any]] = None
    # Checkpointing parameters
    create_checkpoint_callback: Optional[bool] = True
    # Additional exp_manager arguments
    files_to_copy: Optional[List[str]] = None


def exp_manager(trainer: 'pytorch_lightning.Trainer', cfg: Optional[Union[DictConfig, Dict]] = None) -> Path:
    """
    exp_manager is a helper function used to manage folders for experiments. It follows the pytorch lightning paradigm
    of exp_dir/model_or_experiment_name/version. If the lightning trainer has a logger, exp_manager will get exp_dir,
    name, and version from the logger. Otherwise it will use the exp_dir and name arguments to create the logging
    directory. exp_manager also allows for explicit folder creation via explicit_log_dir.
    The version will be a datetime string or an integer. Note, exp_manager does not handle versioning on slurm
    multi-node runs. Datestime version can be disabled if use_datetime_version is set to False.
    It optionally creates TensorBoardLogger, WandBLogger, ModelCheckpoint objects from pytorch lightning. It copies
    sys.argv, and git information if available to the logging directory. It creates a log file for each process to log
    their output into.
    exp_manager additionally has a resume feature which can be used to continuing training from the constructed log_dir.

    Args:
        trainer (pytorch_lightning.Trainer): The lightning trainer.
        cfg (DictConfig, dict): Can have the following keys:
            - explicit_log_dir (str, Path): Can be used to override exp_dir/name/version folder creation. Defaults to
                None, which will use exp_dir, name, and version to construct the logging directory.
            - exp_dir (str, Path): The base directory to create the logging directory. Defaults to None, which logs to
                ./nemo_experiments.
            - name (str): The name of the experiment. Defaults to None which turns into "default" via name = name or
                "default".
            - version (str): The version of the experiment. Defaults to None which uses either a datetime string or
                lightning's TensorboardLogger system of using version_{int}.
            - use_datetime_version (bool): Whether to use a datetime string for version. Defaults to True.
            - resume_if_exists (bool): Whether this experiment is resuming from a previous run. If True, it sets
                trainer.resume_from_checkpoint so that the trainer should auto-resume. exp_manager will move files
                under log_dir to log_dir/run_{int}. Defaults to False.
            - resume_past_end (bool): exp_manager errors out if resume_if_exists is True and a checkpoint matching
                *end.ckpt indicating a previous training run fully completed. This behaviour can be disabled, in which
                case the *end.ckpt will be loaded by setting resume_past_end to True. Defaults to False.
            - resume_ignore_no_checkpoint (bool): exp_manager errors out if resume_if_exists is True and no checkpoint
                could be found. This behaviour can be disabled, in which case exp_manager will print a message and
                continue without restoring, by setting resume_ignore_no_checkpoint to True. Defaults to False.
            - create_tensorboard_logger (bool): Whether to create a tensorboard logger and attach it to the pytorch
                lightning trainer. Defaults to True.
            - summary_writer_kwargs (dict): A dictionary of kwargs that can be passed to lightning's TensorboardLogger
                class. Note that log_dir is passed by exp_manager and cannot exist in this dict. Defaults to None.
            - create_wandb_logger (bool): Whether to create a Weights and Baises logger and attach it to the pytorch
                lightning trainer. Defaults to False.
            - wandb_logger_kwargs (dict): A dictionary of kwargs that can be passed to lightning's WandBLogger
                class. Note that name and project are required parameters if create_wandb_logger is True.
                Defaults to None.
            - create_checkpoint_callback (bool): Whether to create a ModelCheckpoint callback and attach it to the
                pytorch lightning trainer. The ModelCheckpoint saves the top 3 models with the best "val_loss", the most
                recent checkpoint under *last.ckpt, and the final checkpoint after training completes under *end.ckpt.
                Defaults to True.
            - files_to_copy (list): A list of files to copy to the experiment logging directory. Defaults to None which
                copies no files.

    returns:
        log_dir (Path): The final logging directory where logging files are saved. Usually the concatenation of
            exp_dir, name, and version.
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
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg = OmegaConf.merge(schema, cfg)

    error_checks(trainer, cfg)  # Ensures that trainer options are compliant with NeMo and exp_manager arguments

    log_dir, exp_dir, name, version = get_log_dir(
        trainer=trainer,
        exp_dir=cfg.exp_dir,
        name=cfg.name,
        version=cfg.version,
        explicit_log_dir=cfg.explicit_log_dir,
        use_datetime_version=cfg.use_datetime_version,
    )
    if cfg.resume_if_exists:
        check_resume(trainer, log_dir, cfg.resume_past_end, cfg.resume_ignore_no_checkpoint)

    checkpoint_name = name
    # If name returned from get_log_dir is "", use cfg.name for checkpointing
    if checkpoint_name is None or checkpoint_name == '':
        checkpoint_name = cfg.name or "default"
    cfg.name = name  # Used for configure_loggers so that the log_dir is properly set even if name is ""
    cfg.version = version

    # Create the logging directory if it does not exist
    os.makedirs(log_dir, exist_ok=True)  # Cannot limit creation to global zero as all ranks write to own log file
    logging.info(f'Experiments will be logged at {log_dir}')
    trainer._default_root_dir = log_dir

    # Handle Loggers by creating file and handle DEBUG statements
    # Note: trainer.global_rank and trainer.is_global_zero are not set until trainer.fit, so have to hack around it
    global_rank = trainer.node_rank * trainer.num_gpus + trainer.local_rank
    log_file = log_dir / f'nemo_log_globalrank-{global_rank}_localrank-{trainer.local_rank}.txt'
    logging.add_file_handler(log_file)
    logging.rank = global_rank

    # For some reason, LearningRateLogger requires trainer to have a logger. Safer to create logger on all ranks
    # not just global rank 0.
    if cfg.create_tensorboard_logger or cfg.create_wandb_logger:
        configure_loggers(
            trainer,
            exp_dir,
            cfg.name,
            cfg.version,
            cfg.create_tensorboard_logger,
            cfg.summary_writer_kwargs,
            cfg.create_wandb_logger,
            cfg.wandb_logger_kwargs,
        )

    if is_global_rank_zero():
        if cfg.create_checkpoint_callback:
            configure_checkpointing(trainer, log_dir, checkpoint_name)

        # Move files_to_copy to folder and add git information if present
        if cfg.files_to_copy:
            for _file in cfg.files_to_copy:
                copy(Path(_file), log_dir)

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
            f"create_tensorboard_logger: {cfg.create_tensorboard_logger} or create_wandb_logger: {cfg.create_wandb_logger}"
            " was set to True. These can only be used if trainer does not already have a logger."
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


def check_resume(
    trainer: 'pytorch_lightning.Trainer',
    log_dir: str,
    resume_past_end: bool = False,
    resume_ignore_no_checkpoint: bool = False,
):
    """Checks that resume=True was used correctly with the arguments pass to exp_manager. Sets
    trainer.resume_from_checkpoint as necessary.

    Returns:
        log_dir (Path): the log_dir
        exp_dir (str): the base exp_dir without name nor version
        name (str): The name of the experiment
        version (str): The version of the experiment

    Raises:
        NotFoundError: If resume is True, resume_ignore_no_checkpoint is False, and checkpoints could not be found.
        ValueError: If resume is True, and there were more than 1 checkpoint could found.
    """
    if not log_dir:
        raise ValueError(f"Resuming requires the log_dir {log_dir} to be passed to exp_manager")

    checkpoint_dir = Path(Path(log_dir) / "checkpoints")
    checkpoint = None
    end_checkpoints = list(checkpoint_dir.glob("*end.ckpt"))
    last_checkpoints = list(checkpoint_dir.glob("*last.ckpt"))
    if not checkpoint_dir.exists():
        if resume_ignore_no_checkpoint:
            logging.warning(
                f"There was no checkpoint folder at checkpoint_dir :{checkpoint_dir}. Training from scratch."
            )
            return
        else:
            raise NotFoundError(f"There was no checkpoint folder at checkpoint_dir :{checkpoint_dir}. Cannot resume.")
    elif len(end_checkpoints) > 0:
        if resume_past_end:
            if len(end_checkpoints) > 1:
                raise ValueError(f"Multiple multiple checkpoints {end_checkpoints} that matches *end.ckpt.")
            logging.info(f"Resuming from {end_checkpoints[0]}")
            checkpoint = end_checkpoints[0]
        else:
            raise ValueError(
                f"Found {end_checkpoints[0]} indicating that the last training run has already completed."
            )
    elif not len(last_checkpoints) > 0:
        if resume_ignore_no_checkpoint:
            logging.warning(f"There were no checkpoints found in {checkpoint_dir}. Training from scratch.")
            return
        else:
            raise NotFoundError(f"There were no checkpoints found in {checkpoint_dir}. Cannot resume.")
    elif len(last_checkpoints) > 1:
        raise ValueError(f"Multiple multiple checkpoints {last_checkpoints} that matches *last.ckpt.")
    else:
        logging.info(f"Resuming from {last_checkpoints[0]}")
        checkpoint = last_checkpoints[0]

    trainer.resume_from_checkpoint = str(checkpoint)

    if is_global_rank_zero():
        # Check to see if any files exist that need to be moved
        files_to_move = []
        for child in Path(log_dir).iterdir():
            if child.is_file():
                files_to_move.append(child)

        if len(files_to_move) > 0:
            # Move old files to a new folder
            other_run_dirs = Path(log_dir).glob("run_*")
            run_count = 0
            for fold in other_run_dirs:
                if fold.is_dir():
                    run_count += 1
            new_run_dir = Path(Path(log_dir) / f"run_{run_count}")
            new_run_dir.mkdir()
            for _file in files_to_move:
                move(str(_file), str(new_run_dir))


def check_explicit_log_dir(
    trainer: 'pytorch_lightning.Trainer', explicit_log_dir: [Path, str], exp_dir: str, name: str, version: str
) -> (Path, str, str, str):
    """ Checks that the passed arguments are compatible with explicit_log_dir.

    Returns:
        log_dir (Path): the log_dir
        exp_dir (str): the base exp_dir without name nor version
        name (str): The name of the experiment
        version (str): The version of the experiment

    Raise:
        LoggerMisconfigurationError
    """
    if trainer.logger is not None:
        raise LoggerMisconfigurationError(
            "The pytorch lightning trainer that was passed to exp_manager contained a logger and explicit_log_dir: "
            f"{explicit_log_dir} was pass to exp_manager. Please remove the logger from the lightning trainer."
        )
    if exp_dir or name or version:
        logging.error(
            f"exp_manager received explicit_log_dir: {explicit_log_dir} and at least one of exp_dir: {exp_dir}, "
            f"name: {name}, or version: {version}. Please note that exp_dir, name, and version will be ignored."
        )
    if is_global_rank_zero() and Path(explicit_log_dir).exists():
        logging.warning(f"Exp_manager is logging to {explicit_log_dir}, but it already exists.")
    return Path(explicit_log_dir), str(explicit_log_dir), "", ""


def get_log_dir(
    trainer: 'pytorch_lightning.Trainer',
    exp_dir: str = None,
    name: str = None,
    version: str = None,
    explicit_log_dir: str = None,
    use_datetime_version: bool = True,
) -> (Path, str, str, str):
    """
    Obtains the log_dir used for exp_manager.

    Returns:
        log_dir (Path): the log_dir
        exp_dir (str): the base exp_dir without name nor version
        name (str): The name of the experiment
        version (str): The version of the experiment

    Raise:
        LoggerMisconfigurationError: If trainer is incompatible with arguments
        NotFoundError: If resume is True, resume_ignore_no_checkpoint is False, and checkpoints could not be found.
        ValueError: If resume is True, and there were more than 1 checkpoint could found.
    """
    if explicit_log_dir:  # If explicit log_dir was passed, short circuit
        return check_explicit_log_dir(trainer, explicit_log_dir, exp_dir, name, version)

    # Default exp_dir to ./nemo_experiments if None was passed
    _exp_dir = exp_dir
    if exp_dir is None:
        _exp_dir = str(Path.cwd() / 'nemo_experiments')

    # If the user has already defined a logger for the trainer, use the logger defaults for logging directory
    if trainer.logger is not None:
        if trainer.logger.save_dir:
            if exp_dir:
                raise LoggerMisconfigurationError(
                    "The pytorch lightning trainer that was passed to exp_manager contained a logger, the logger's "
                    f"save_dir was not None, and exp_dir ({exp_dir}) was not None. If trainer.logger.save_dir "
                    "exists, exp_manager will use trainer.logger.save_dir as the logging directory and exp_dir "
                    "must be None."
                )
            _exp_dir = trainer.logger.save_dir
        if name:
            raise LoggerMisconfigurationError(
                "The pytorch lightning trainer that was passed to exp_manager contained a logger, and name: "
                f"{name} was also passed to exp_manager. If the trainer contains a "
                "logger, exp_manager will use trainer.logger.name, and name passed to exp_manager must be None."
            )
        name = trainer.logger.name
        version = f"version_{trainer.logger.version}"
    # Use user-defined exp_dir, project_name, exp_name, and versioning options
    else:
        name = name or "default"
        version = version or os.environ.get(NEMO_ENV_VARNAME_VERSION, None)

        if version is None:
            if trainer.is_slurm_managing_tasks:
                logging.warning("Running on a slurm cluster. exp_manager will not add a version number.")
                version = ""
            elif is_global_rank_zero():
                if use_datetime_version:
                    version = time.strftime('%Y-%m-%d_%H-%M-%S')
                else:
                    tensorboard_logger = TensorBoardLogger(save_dir=Path(_exp_dir), name=name, version=version)
                    version = f"version_{tensorboard_logger.version}"
                os.environ[NEMO_ENV_VARNAME_VERSION] = version

    log_dir = Path(_exp_dir) / Path(str(name)) / Path(str(version))
    return log_dir, str(_exp_dir), name, version


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
    """ A thin wrapper on Lightning's LoggerCollection such that name and version are better aligned with exp_manager
    """

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
    trainer: 'pytorch_lightning.Trainer',
    exp_dir: [Path, str],
    name: str,
    version: str,
    create_tensorboard_logger: bool,
    summary_writer_kwargs: dict,
    create_wandb_logger: bool,
    wandb_kwargs: dict,
):
    """ Creates TensorboardLogger and/or WandBLogger and attach them to trainer. Raises ValueError if
    summary_writer_kwargs or wandb_kwargs are misconfigured.
    """
    # Potentially create tensorboard logger and/or WandBLogger
    logger_list = []
    if create_tensorboard_logger:
        if summary_writer_kwargs is None:
            summary_writer_kwargs = {}
        elif "log_dir" in summary_writer_kwargs:
            raise ValueError(
                "You cannot pass `log_dir` as part of `summary_writer_kwargs`. `log_dir` is handled by lightning's "
                "TensorBoardLogger logger."
            )
        tensorboard_logger = TensorBoardLogger(save_dir=exp_dir, name=name, version=version, **summary_writer_kwargs)
        logger_list.append(tensorboard_logger)
        logging.info("TensorboardLogger has been set up")

    if create_wandb_logger:
        if wandb_kwargs is None:
            wandb_kwargs = {}
        if "name" not in wandb_kwargs and "project" not in wandb_kwargs:
            raise ValueError("name and project are required for wandb_logger")
        wandb_logger = WandbLogger(save_dir=exp_dir, version=version, **wandb_kwargs)

        logger_list.append(wandb_logger)
        logging.info("WandBLogger has been set up")

    logger_list = (
        LoggerList(logger_list, nemo_name=name, nemo_version=version) if len(logger_list) > 1 else logger_list[0]
    )
    trainer.configure_logger(logger_list)


def configure_checkpointing(trainer: 'pytorch_lightning.Trainer', log_dir: Path, name: str):
    """ Adds ModelCheckpoint to trainer. Raises CheckpointMisconfigurationError if trainer already has a ModelCheckpoint
    callback or if trainer.weights_save_path was passed to Trainer.
    """
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
            # TODO: Remove try, except block once lightning's ModelCheckpoint is stable
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
