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
from pathlib import Path
from shutil import copyfile

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from nemo.utils import logging


def exp_manager(
    trainer,
    root_dir=None,
    name=None,
    create_tensorboard_logger=True,
    create_checkpoint_callback=True,
    files_to_copy=None,
):
    """
    ExpManager is a helper function used to manage folders for experiments. It follows the pytorch lightning paradigm of
    root_dir/model_or_experiment_name/version. It optionally creates TensorBoardLogger, and ModelCheckpoint objects
    from pytorch lightning.
    """
    # TODO: Print a warning message if user is not running ddp / slurm since we test on those
    # Default root_dir to ~/NeMo_experiments if None was passed
    _root_dir = root_dir
    if root_dir is None:
        _root_dir = str(Path.home() / 'NeMo_experiments')
    # If the user has already defined a logger for the trainer, use the logger defaults for logging directory
    if trainer.logger is not None:
        if trainer.logger.save_dir:
            if root_dir:
                raise ValueError(
                    "The pytorch_lightning trainer that was passed to ExpManager contained a logger, the logger's "
                    f"save_dir was not None, and root_dir ({root_dir}) was not None. If trainer.logger.save_dir "
                    "exists, ExpManager will use trainer.logger.save_dir as the logging directory and root_dir "
                    "must be None."
                )
            _root_dir = trainer.logger.save_dir
        if name:
            raise ValueError(
                f"The pytorch_lightning trainer that was passed to ExpManager contained a logger, and name ({name})"
                " was also passed to ExpManager. If the trainer contains a logger, ExpManager will use "
                "trainer.logger.name, and name that is passed to ExpManager must be None."
            )
        if create_tensorboard_logger:
            raise ValueError(
                "The pytorch_lightning trainer that was passed to ExpManager contained a logger, and "
                "create_tensorboard_logger was set to True. create_tensorboard_logger can only be used if trainer "
                "does not already have a logger."
            )
        name = trainer.logger.name
        version = trainer.logger.version
    else:
        version = None
        if trainer.is_slurm_managing_tasks:
            logging.warning("Running on a slurm cluster. Versioning by datetime will not work.")
        elif trainer.is_global_zero():
            version = time.strftime('%Y-%m-%d_%H-%M-%S')
        else:
            pass  # TODO: grab tm_suffix from sys.argv

        if name is None:
            name = "default"

    # Always create TensorBoardLogger, so we can retrieve version if running on slurm
    tensorboard_logger = TensorBoardLogger(save_dir=_root_dir, name=name, version=version)
    if create_tensorboard_logger:
        # Attach logger to trainer
        trainer.configure_logger(tensorboard_logger)
    if version is None:
        version = tensorboard_logger.version

    if create_checkpoint_callback:
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                raise ValueError(
                    "The pytorch_lightning trainer that was passed to ExpManager contained a ModelCheckpoint "
                    "and create_checkpoint_callback was set to True. Please either set create_checkpoint_callback "
                    "to False, or remove ModelCheckpoint from the lightning trainer"
                )
        checkpoint_callback = ModelCheckpoint(save_top_k=3, save_last=True)
        trainer.callbacks.append(checkpoint_callback)

    # Create the logging directory if it does not exist
    log_dir = Path(_root_dir, name, version)
    os.makedirs(log_dir, exist_ok=True)  # Cannot limit creation to global zero as all ranks write to own log file

    # Move files_to_copy to folder and add git information if present
    if trainer.is_global_zero():
        if files_to_copy:
            for _file in files_to_copy:
                basename = os.path.basename(_file)
                basename, ending = os.path.splitext(basename)
                basename = basename + f"_{version}" + ending
                copyfile(_file, log_dir / basename)

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
    log_file = log_dir / f'log_globalrank-{trainer.global_rank}_localrank-{trainer.local_rank}.txt'
    logging.add_file_handler(log_file)
    logging.rank = trainer.global_rank


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
