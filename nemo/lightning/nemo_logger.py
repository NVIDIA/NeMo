import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Type, Union

import lightning_fabric as fl
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint as PTLModelCheckpoint

from nemo.constants import NEMO_ENV_VARNAME_TESTING, NEMO_ENV_VARNAME_VERSION
from nemo.lightning.pytorch.callbacks import ModelCheckpoint, ModelCheckpointParams
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.env_var_parsing import get_envbool
from nemo.utils.exp_manager import (
    check_explicit_log_dir,
    CheckpointMisconfigurationError
)
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.mcore_logger import add_handlers_to_mcore_logger

@dataclass
class NeMoLogger:
    name: str = "default"
    dir: Optional[str] = None
    explicit_log_dir: Optional[str] = None
    version: Optional[str] = None
    use_datetime_version: bool = True
    log_local_rank_0_only: bool = False
    log_global_rank_0_only: bool = False
    files_to_copy: Optional[List[str]] = None
    update_logger_directory: bool = True

    def __post_init__(self):
        if self.log_local_rank_0_only is True and self.log_global_rank_0_only is True:
            raise ValueError(
                f"Cannot set both log_local_rank_0_only and log_global_rank_0_only to True. Please set either one or neither."
            )

    def setup(
        self,
        trainer: Union[pl.Trainer, fl.Fabric],
        resume_if_exists: bool = False,
        model_checkpoint_cls: Type[ModelCheckpoint] = None, ## optional checkpoint callback to instantiate and add to the trainer
        model_checkpoint_params: Optional[ModelCheckpointParams] = {},
    ):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = trainer.node_rank * trainer.world_size + local_rank
        logging.rank = global_rank

        if self.explicit_log_dir and isinstance(trainer, pl.Trainer):  # If explicit log_dir was passed, short circuit
            return check_explicit_log_dir(trainer, self.explicit_log_dir, self.dir, self.name, self.version)

        # Default dir to ./nemo_experiments if None was passed
        _dir = self.dir
        if self.dir is None:
            _dir = str(Path.cwd() / 'nemo_experiments')

        if not self.name:
            self.name = "default"

        if isinstance(trainer, pl.Trainer) and trainer.logger is not None:
            if self.update_logger_directory:
                logging.warning(
                    f'"update_logger_directory" is True. Overwriting logger "save_dir" to {_dir} and "name" to {self.name}'
                )
                trainer.logger._root_dir = _dir
                trainer.logger._name = self.name

        version = self.version or os.environ.get(NEMO_ENV_VARNAME_VERSION, None)
        if is_global_rank_zero():
            if self.use_datetime_version:
                version = time.strftime('%Y-%m-%d_%H-%M-%S')
        if resume_if_exists:
            logging.warning(
                "No version folders would be created under the log folder as 'resume_if_exists' is enabled."
            )
            version = None
        if version:
            if is_global_rank_zero():
                os.environ[NEMO_ENV_VARNAME_VERSION] = version

        log_dir = Path(_dir) / Path(str(self.name)) / Path("" if version is None else str(version))
        # update app_state with log_dir, exp_dir, etc
        app_state = AppState()
        app_state.log_dir = log_dir
        app_state.exp_dir = _dir
        app_state.name = self.name
        app_state.version = version

        os.makedirs(log_dir, exist_ok=True)  # Cannot limit creation to global zero as all ranks write to own log file
        logging.info(f'Experiments will be logged at {log_dir}')

        if model_checkpoint_cls is not None:
            for callback in trainer.callbacks:
                print(f'CALLBACK: {callback}')
                if isinstance(callback, PTLModelCheckpoint):
                    raise CheckpointMisconfigurationError(
                        "The pytorch lightning trainer that was passed contained a ModelCheckpoint "
                        "and model_checkpoint_cls was not None. Please either set model_checkpoint_cls "
                        "to None, or remove ModelCheckpoint from the lightning trainer"
                    )
                if "val" in model_checkpoint_params.monitor:
                    if (
                        trainer.max_epochs is not None
                        and trainer.max_epochs != -1
                        and trainer.max_epochs < trainer.check_val_every_n_epoch
                    ):
                        logging.error(
                            "The checkpoint callback was told to monitor a validation value but trainer.max_epochs("
                            f"{trainer.max_epochs}) was less than trainer.check_val_every_n_epoch({trainer.check_val_every_n_epoch}"
                            f"). It is very likely this run will fail with ModelCheckpoint(monitor='{params.monitor}') not found "
                            "in the returned metrics. Please ensure that validation is run within trainer.max_epochs."
                        )
                    elif trainer.max_steps is not None and trainer.max_steps != -1:
                        logging.warning(
                            "The checkpoint callback was told to monitor a validation value and trainer's max_steps was set to "
                            f"{trainer.max_steps}. Please ensure that max_steps will run for at least "
                            f"{trainer.check_val_every_n_epoch} epochs to ensure that checkpointing will not error out."
                        )

            checkpoint_callback = model_checkpoint_cls(**asdict(model_checkpoint_params))
            trainer.callbacks.append(checkpoint_callback)

        if isinstance(trainer, pl.Trainer):
            for callback in trainer.callbacks:
                if isinstance(callback, PTLModelCheckpoint):
                    ## TODO: make configurable
                    callback.dirpath = Path(log_dir / "checkpoints")  # app_state.exp_dir
                    if callback.filename is None:
                        callback.filename = f'{self.name}--{{{callback.monitor}:.4f}}-{{epoch}}'
                    ModelCheckpoint.CHECKPOINT_NAME_LAST = callback.filename + '-last'

        # This is set if the env var NEMO_TESTING is set to True.
        nemo_testing = get_envbool(NEMO_ENV_VARNAME_TESTING, False)

        # Handle logging to file
        log_file = log_dir / f'nemo_log_globalrank-{global_rank}_localrank-{local_rank}.txt'
        if self.log_local_rank_0_only is True and not nemo_testing:
            if local_rank == 0:
                logging.add_file_handler(log_file)
        elif self.log_global_rank_0_only is True and not nemo_testing:
            if global_rank == 0:
                logging.add_file_handler(log_file)
        else:
            # Logs on all ranks.
            logging.add_file_handler(log_file)

        add_handlers_to_mcore_logger()

        app_state.files_to_copy = self.files_to_copy
        app_state.cmd_args = sys.argv

        return app_state

    def teardown(self):
        pass

