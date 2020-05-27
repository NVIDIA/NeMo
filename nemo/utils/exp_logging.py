# Copyright (c) 2019 NVIDIA Corporation
import os
import subprocess
import sys
import time
from shutil import copyfile

from nemo.utils import logging
from nemo.utils.decorators import deprecated


# from nemo.utils import logging
@deprecated(
    version=0.11,
    explanation=(
        "Please use nemo.logging instead by using from nemo.utils import logging and logging.info(), "
        "logging.warning() , etc."
    ),
)
def get_logger(unused):
    return logging


# class ContextFilter(logging.Filter):
#     """
#     This is a filter which injects contextual information into the log.
#     Use it when we want to inject worker number into the log message.

#     Usage:
#     logger = get_logger(name)
#     tmp = logging.Formatter(
#         'WORKER %(local_rank)s: %(asctime)s - %(levelname)s - %(message)s')
#     logger.addFilter(ContextFilter(self.local_rank))

#     """

#     def __init__(self, local_rank):
#         super().__init__()
#         self.local_rank = local_rank

#     def filter(self, record):
#         record.local_rank = self.local_rank
#         return True


class ExpManager:
    """ Note: Users should not have to call ExpManager as it is done
    automically inside NeuralFactory. Not all defaults match NeuralFactory
    defaults.
    ExpManager helps create a work directory used to log experiment
    files. It additionally creates a checkpoint directory, tensboard
    directory, tensorboardX.SummaryWriter object,
    copies any files passed with files_to_copy to the work
    directory, and creates loggers used to print to screen and file.

    Args:
    work_dir (str): Directory that Expmanager should either create or
        save log files and directories to.
        Defaults to None and does not create a work directory.
    local_rank (int): None for single-gpu, else the local id for distributed
        setups.
        Defaults to None
    global_rank (int): None for single-gpu, else the global id for distributed
        setups.
        Defaults to None
    use_tb (bool): Whether to create a tensorboardX.SummaryWriter object
        Defaults to True.
    exist_ok (bool): If False, ExpManager will crash if the work_dir
        already exists. Must be false for distributed runs.
    ckpt_dir (str). Whether to create a subdir called ckpt_dir.
        Defaults to which creates a subdir called "checkpoints".
    tb_dir (str): Whether to create a subdir called tb_dir.
        Defaults to which creates a subdir called "tensorboard".
    files_to_copy (list): Should be a list of paths of files that you
        want to copy to work_dir. Useful for copying configs and model
        scrtips.
        Defaults to None which copies no files.
    add_time (bool): Whether to add a datetime ending to work_dir
        Defaults to True.
    broadcast_func (func): Only required if add_time is True and
        distributed. broadcast_func should accept a string that contains
        the datetime suffix such that all ranks are consistent on work_dir
        name.
    """

    def __init__(
        self,
        work_dir=None,
        local_rank=None,
        global_rank=None,
        use_tb=True,
        exist_ok=True,
        ckpt_dir=None,
        tb_dir=None,
        files_to_copy=None,
        add_time=True,
        broadcast_func=None,
    ):
        self.local_rank = local_rank if local_rank is not None else 0
        self.global_rank = global_rank if global_rank is not None else 0
        self.log_file = None
        self.tb_writer = None
        self.work_dir = None
        # Note that if ckpt_dir is None, the default behaviour is to put it
        # under {self.work_dir}/checkpoints
        # If both ckpt_dir and work_dir are None, no ckpt_dir is created
        self.ckpt_dir = None
        # tb_dir behaves the same as ckpt_dir except default folder name is
        # tensorboard instead of checkpoints
        self.tb_dir = tb_dir
        tm_suf = time.strftime('%Y-%m-%d_%H-%M-%S')
        if global_rank is not None and add_time:
            if broadcast_func is None:
                raise ValueError(
                    "local rank was not None, but ExpManager was not passed a "
                    "broadcast function to broadcast the datetime suffix"
                )
            if global_rank == 0:
                broadcast_func(string=tm_suf)
            else:
                tm_suf = broadcast_func(str_len=len(tm_suf))

        # Create work_dir if specified
        if work_dir:
            self.work_dir = work_dir
            # only create tm_sur dir if checkpoints dir is not present in the work_dir
            if add_time:
                self.work_dir = os.path.join(work_dir, tm_suf)
            self.make_dir(self.work_dir, exist_ok)
            self.ckpt_dir = os.path.join(self.work_dir, 'checkpoints')

            if use_tb:
                self.get_tb_writer(exist_ok=exist_ok)

            if files_to_copy and self.global_rank == 0:
                for file in files_to_copy:
                    basename = os.path.basename(file)
                    basename, ending = os.path.splitext(basename)
                    basename = basename + f"_{tm_suf}" + ending
                    copyfile(file, os.path.join(self.work_dir, basename))
            if self.global_rank == 0:
                # Create files for cmd args and git info
                with open(os.path.join(self.work_dir, f'cmd-args_{tm_suf}.log'), 'w') as f:
                    f.write(" ".join(sys.argv))

                # Try to get git hash
                git_repo, git_hash = get_git_hash()
                if git_repo:
                    git_log_file = os.path.join(self.work_dir, f'git-info_{tm_suf}.log')
                    with open(git_log_file, 'w') as f:
                        f.write(f'commit hash: {git_hash}')
                        f.write(get_git_diff())

        # Create loggers
        if bool(work_dir):
            self.add_file_handler_to_logger()
        if use_tb and not work_dir:
            raise ValueError("ExpManager received use_tb as True but did not receive a work_dir")

        if ckpt_dir:
            self.ckpt_dir = ckpt_dir
        if self.ckpt_dir:
            self.make_dir(self.ckpt_dir, exist_ok)

    def add_file_handler_to_logger(self):
        self.log_file = f'{self.work_dir}/log_globalrank-{self.global_rank}_' f'localrank-{self.local_rank}.txt'
        logging.add_file_handler(self.log_file)

    def make_dir(self, dir_, exist_ok):
        # We might want to limit folder creation to only global_rank 0
        os.makedirs(dir_, exist_ok=exist_ok)

    def get_tb_writer(self, tb_dir=None, exist_ok=True):
        if self.global_rank == 0:
            if tb_dir is None:
                if not hasattr(self, 'tb_dir') or self.tb_dir is None:
                    self.tb_dir = os.path.join(self.work_dir, 'tensorboard')
            else:  # if user passes in tb_dir then override the exp's tb_dir
                self.tb_dir = tb_dir
            self.make_dir(self.tb_dir, exist_ok)

            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb_writer = SummaryWriter(self.tb_dir)
            except ImportError:
                self.tb_writer = None
                logging.info('Not using TensorBoard.')
                logging.info('Install tensorboardX to use TensorBoard')
        return self.tb_writer

    def log_exp_info(self, params, print_everywhere=False):
        if print_everywhere or self.global_rank == 0:
            logging.info("NEMO MODEL'S PARAMETERS")
            for key in params:
                logging.info(f'{key}\t{params[key]}')
            logging.info(f'Experiment output is stored in {self.work_dir}')


def get_git_hash():
    try:
        return (
            True,
            subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode(),
        )
    except subprocess.CalledProcessError as e:
        return False, "{}\n".format(e.output.decode("utf-8"))


def get_git_diff():
    try:
        return subprocess.check_output(['git', 'diff'], stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        return "{}\n".format(e.output.decode("utf-8"))
