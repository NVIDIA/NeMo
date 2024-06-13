import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import fiddle as fdl
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint as PTLModelCheckpoint
from typing_extensions import Self

from nemo.constants import NEMO_ENV_VARNAME_TESTING, NEMO_ENV_VARNAME_VERSION
from nemo.lightning.io.mixin import IOMixin
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.resume import Resume
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.env_var_parsing import get_envbool
from nemo.utils.exp_manager import LoggerMisconfigurationError, check_explicit_log_dir
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.mcore_logger import add_handlers_to_mcore_logger


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
        Sets the log_dir and exp_dir used for the experiment.

        Returns:
            app_state (AppState): AppState that stores the following attributes
            log_dir (Path): the log_dir
            exp_dir (str): the base exp_dir without name nor version
            name (str): The name of the experiment
            version (str): The version of the experiment

        Raise:
            LoggerMisconfigurationError: If trainer is incompatible with arguments
            ValueError: If log_local_rank_0_only and log_global_rank_0_only are both True
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

        if not name:
            name = "default"

        if self.logger is not None:
            if update_logger_directory:
                logging.warning(
                    f'"update_logger_directory" is True. Overwriting logger "save_dir" to {exp_dir} and "name" to {name}'
                )
                self.logger._root_dir = exp_dir
                self.logger._name = name

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
                callback.dirpath = Path(log_dir / "checkpoints")  # app_state.exp_dir
                if callback.filename is None:
                    callback.filename = f'{name}--{{{callback.monitor}:.4f}}-{{epoch}}'
                if callback.prefix is None:
                    callback.prefix = name
                ModelCheckpoint.CHECKPOINT_NAME_LAST = callback.filename + '-last'

        if log_local_rank_0_only is True and log_global_rank_0_only is True:
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

        app_state.files_to_copy = files_to_copy
        app_state.cmd_args = sys.argv
        return app_state
