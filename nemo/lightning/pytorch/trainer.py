from copy import deepcopy
import time

import fiddle as fdl
import pytorch_lightning as pl
from typing_extensions import Self

from nemo.lightning.io.mixin import IOMixin

import os
from pathlib import Path
from typing import Optional, List
import sys
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.env_var_parsing import get_envbool
from nemo.utils.exp_manager import (
    check_explicit_log_dir,
    LoggerMisconfigurationError,
    _filter_out_unfinished_checkpoints,
    NotFoundError
)
from nemo.utils.mcore_logger import add_handlers_to_mcore_logger
from nemo.utils.get_rank import is_global_rank_zero
## circular import
#from nemo.lightning import ModelCheckpoint
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint as PTLModelCheckpoint

NEMO_ENV_VARNAME_VERSION="NEMO_EXPM_VERSION"
NEMO_ENV_VARNAME_TESTING="NEMO_TESTING"

class Resume:
    def nemo_path(self, model) -> Optional[Path]:
        raise NotImplementedError


class AutoResume(Resume):
    def __init__(
        self,
        path: Optional[str] = None, ## old resume_from_checkpoint
        dirpath: Optional[str] = None, ## optional path to checkpoint directory
        ## make this fn more clear
        import_path: Optional[str] = None, ## old dirpath ## what is this?
        #log_dir: str = None,

        resume_if_exists: bool = False,
        resume_past_end: bool = False,
        resume_ignore_no_checkpoint: bool = False,
    ):
        if path and import_path:
            raise ValueError("Only one of path or import_path can be set")

        self.path = path
        self.dirpath = dirpath
        self.import_path = import_path
        #self.log_dir = log_dir ## TODO: don't set this here.. this should be inferred from AppState!
        self.resume_if_exists = resume_if_exists
        self.resume_past_end = resume_past_end
        self.resume_ignore_no_checkpoint = resume_ignore_no_checkpoint

    def nemo_path(self, model=None) -> Optional[Path]:

        if self.import_path:
            return model.import_ckpt(self.import_path)

        ### refactored from exp_manager
        checkpoint = None
        log_dir = AppState().log_dir
        if self.path:
            checkpoint = self.path
        if self.resume_if_exists:
            # Use <log_dir>/checkpoints/ unless `dirpath` is set
            checkpoint_dir = Path(self.dirpath) if self.dirpath else Path(Path(log_dir) / "checkpoints")

            # when using distributed checkpointing, checkpoint_dir is a directory of directories
            # we check for this here
            dist_checkpoints = [d for d in list(checkpoint_dir.glob("*")) if d.is_dir()]
            end_dist_checkpoints = [d for d in dist_checkpoints if d.match("*end")]
            last_dist_checkpoints = [d for d in dist_checkpoints if d.match("*last")]

            end_checkpoints = _filter_out_unfinished_checkpoints(end_dist_checkpoints)
            last_checkpoints = _filter_out_unfinished_checkpoints(last_dist_checkpoints)

            if not checkpoint_dir.exists() or (not len(end_checkpoints) > 0 and not len(last_checkpoints) > 0):
                if self.resume_ignore_no_checkpoint:
                    warn = f"There were no checkpoints found in checkpoint_dir or no checkpoint folder at checkpoint_dir :{checkpoint_dir}. "
                    if checkpoint is None:
                        warn += "Training from scratch."
                    elif checkpoint == resume_from_checkpoint:
                        warn += f"Training from {resume_from_checkpoint}."
                    logging.warning(warn)
                else:
                    raise NotFoundError(
                        f"There were no checkpoints found in checkpoint_dir or no checkpoint folder at checkpoint_dir :{checkpoint_dir}. Cannot resume."
                    )
            elif len(end_checkpoints) > 0:
                if resume_past_end:
                    if len(end_checkpoints) > 1:
                        if 'mp_rank' in str(end_checkpoints[0]):
                            checkpoint = end_checkpoints[0]
                        else:
                            raise ValueError(f"Multiple checkpoints {end_checkpoints} that matches *end.ckpt.")
                else:
                    raise ValueError(
                        f"Found {end_checkpoints[0]} indicating that the last training run has already completed."
                    )
            elif len(last_checkpoints) > 1:
                if any([s for s in ['mp_rank', 'tp_rank', 'fsdp_shard'] if s in str(last_checkpoints[0])]):
                    checkpoint = last_checkpoints[0]
                    checkpoint = uninject_model_parallel_rank(checkpoint)
                else:
                    raise ValueError(f"Multiple checkpoints {last_checkpoints} that matches *last.ckpt.")
            else:
                checkpoint = last_checkpoints[0]

        return str(checkpoint) if checkpoint is not None else None

class Trainer(pl.Trainer, IOMixin):
    def io_init(self, **kwargs) -> fdl.Config[Self]:
        # Each argument of the trainer can be stateful so we copy them
        cfg_kwargs = {k: deepcopy(v) for k, v in kwargs.items()}

        return fdl.Config(type(self), **cfg_kwargs)

    def setup_nemo(
        self,
        resume: Resume,
        exp_dir: Optional[str] = None,
        explicit_log_dir: Optional[str] = None,
        name: Optional[str] = None,
        version: Optional[str] = None,
        use_datetime_version: bool = True,
        log_local_rank_0_only: bool = False,
        log_global_rank_0_only: bool = False,
        files_to_copy: Optional[List[str]] = None,
        update_logger_directory: bool = True,
    ):
        """
        TODO: update docstring!!!
        Obtains the log_dir used for exp_manager.

        Returns:
            log_dir (Path): the log_dir
            exp_dir (str): the base exp_dir without name nor version
            name (str): The name of the experiment
            version (str): The version of the experiment
            explicit_log_dir (str): The explicit path to the log folder. Defaults to False.
            use_datetime_version (bool): Uses date and time as the version of the log folder. Defaults to True.
                version folders would not get created.

        Raise:
            LoggerMisconfigurationError: If trainer is incompatible with arguments
            NotFoundError: If resume is True, resume_ignore_no_checkpoint is False, and checkpoints could not be found.
            ValueError: If resume is True, and there were more than 1 checkpoint could found.
        """
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = self.node_rank * self.num_devices + local_rank
        logging.rank = global_rank

        if explicit_log_dir:  # If explicit log_dir was passed, short circuit
            return check_explicit_log_dir(self, explicit_log_dir, exp_dir, name, version)

        # Default exp_dir to ./nemo_experiments if None was passed
        _exp_dir = exp_dir
        if exp_dir is None:
            _exp_dir = str(Path.cwd() / 'nemo_experiments')

        # If the user has defined a logger for the trainer, use the logger defaults for logging directory
        ## TODO: I don't think we want this anymore
        ## since before loggers were created after creating log dir
        '''if self.logger is not None:
            if update_logger_directory:
                logging.warning(f'"update_logger_directory" is True. Overwriting logger "save_dir" to write to {exp_dir}')
                self.logger._root_dir = exp_dir

            if name:
                raise LoggerMisconfigurationError(
                    "The pytorch lightning trainer contains a logger, and name: "
                    f"{name} was also passed to setup_nemo. If the trainer contains a "
                    "logger, setup_nemo will use trainer.logger.name, and name passed to setup_nemo must be None."
                )
            name = self.logger.name'''
        if not name:
            name = "default"

        version = version or os.environ.get(NEMO_ENV_VARNAME_VERSION, None)
        if is_global_rank_zero():
            if use_datetime_version:
                version = time.strftime('%Y-%m-%d_%H-%M-%S')
        if resume.resume_if_exists:
                logging.warning(
                    "No version folders would be created under the log folder as 'resume_if_exists' is enabled."
                )
                version = None
        if version:
            if is_global_rank_zero():
                os.environ[NEMO_ENV_VARNAME_VERSION] = version

        log_dir = Path(_exp_dir) / Path(str(name)) / Path("" if version is None else str(version))
        # update app_state with log_dir, exp_dir, etc
        app_state = AppState()
        app_state.log_dir = log_dir
        app_state.exp_dir = exp_dir
        app_state.name = name
        app_state.version = version
        #app_state.checkpoint_name = checkpoint_name
        #app_state.create_checkpoint_callback = cfg.create_checkpoint_callback

        # Create the logging directory if it does not exist
        os.makedirs(log_dir, exist_ok=True)  # Cannot limit creation to global zero as all ranks write to own log file
        logging.info(f'Experiments will be logged at {log_dir}')
        self._default_root_dir = log_dir

        ckpt_path = resume.nemo_path()
        if ckpt_path:
            self.ckpt_path = ckpt_path

        ## TODO: assert that we only have one checkpoint callback?
        for callback in self.callbacks:
            if isinstance(callback, PTLModelCheckpoint):
                ## TODO: make configurable
                callback.dirpath = Path(log_dir / "checkpoints") #app_state.exp_dir
                ## Adding "epoch" to the checkpoint name surfaces a bug in the 
                ## ModelCheckpoint class in which the final checkpoint attempts to be saved twice
                ## with two different epochs
                callback.filename=f'{name}--{{{callback.monitor}:.4f}}-{{epoch}}-{{step}}'
                #callback.filename=f'{name}--{{{callback.monitor}:.4f}}-{{step}}'
                if callback.prefix is None:
                    callback.prefix = name
                ModelCheckpoint.CHECKPOINT_NAME_LAST = callback.filename + '-last'

        if log_local_rank_0_only is True and cfg.log_global_rank_0_only is True:
            raise ValueError(
                f"Cannot set both log_local_rank_0_only and log_global_rank_0_only to True. Please set either one or neither."
            )

        # This is set if the env var NEMO_TESTING is set to True.
        nemo_testing = get_envbool(NEMO_ENV_VARNAME_TESTING, False)

        # Handle logging to file
        log_file = log_dir / f'nemo_log_globalrank-{global_rank}_localrank-{local_rank}.txt'
        if log_local_rank_0_only is True and not nemo_testing:
            if local_rank == 0:
                logging.add_file_handler(log_file)
        elif log_global_rank_0_only is True and not nemo_testing:
            if global_rank == 0:
                logging.add_file_handler(log_file)
        else:
            # Logs on all ranks.
            logging.add_file_handler(log_file)

        add_handlers_to_mcore_logger()

        AppState.files_to_copy = files_to_copy
        AppState.cmd_args = sys.argv

        return AppState
