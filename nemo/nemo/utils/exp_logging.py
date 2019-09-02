import logging
import os
import time
from shutil import copyfile


loggers = {}


class ContextFilter(logging.Filter):
    """
    This is a filter which injects contextual information into the log.
    Use it when we want to inject worker number into the log message.

    Usage:
    logger = get_logger(name)
    tmp = logging.Formatter(
        'WORKER %(local_rank)s: %(asctime)s - %(levelname)s - %(message)s')
    logger.addFilter(ContextFilter(self.local_rank))

    """

    def __init__(self, local_rank):
        super().__init__()
        self.local_rank = local_rank

    def filter(self, record):
        record.local_rank = self.local_rank
        return True


def get_logger(name):
    """ A wrapper function around logging.getLogger
    to ensure that we don't create duplicate loggers
    """
    global loggers

    if name not in loggers:
        loggers[name] = logging.getLogger(name)

    return loggers[name]


def copy_wo_overwrite(dir_, file_to_copy):
    basename = os.path.basename(file_to_copy)
    i = 0
    basename, ending = os.path.splitext(basename)
    basename = basename + "_run{}" + ending
    while True:
        if os.path.isfile(
                os.path.join(dir_, basename.format(i))):
            i += 1
            continue
        else:
            copyfile(file_to_copy,
                     os.path.join(dir_, basename.format(i)))
            break


class ExpManager:
    def __init__(
            self,
            work_dir=None,
            local_rank=None,
            use_tb=True,
            exist_ok=True,
            ckpt_dir=None,
            tb_dir=None,
            files_to_copy=None,
            add_time=True):
        self.local_rank = local_rank if local_rank is not None else 0
        self.logger = None
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

        # Create work_dir if specified
        if work_dir:
            self.work_dir = work_dir
            if add_time:
                self.work_dir = os.path.join(
                    work_dir, time.strftime('%Y%m%d-%H%M%S'))
            self.make_dir(self.work_dir, exist_ok)
            if use_tb:
                self.get_tb_writer(exist_ok=exist_ok)
            self.ckpt_dir = f'{self.work_dir}/checkpoints'
            if files_to_copy:
                for file in files_to_copy:
                    copy_wo_overwrite(self.work_dir, file)

        # Create loggers
        self.create_logger(log_file=bool(work_dir))
        if use_tb and not work_dir:
            raise ValueError("ExpManager received use_tb as True but did not "
                             "receive a work_dir")

        if ckpt_dir:
            self.ckpt_dir = ckpt_dir
        if self.ckpt_dir:
            self.make_dir(self.ckpt_dir, exist_ok)

    def create_logger(self, name='', level=logging.INFO, log_file=True):
        logger = get_logger(name)
        tmp = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        if self.local_rank == 0:
            logger.setLevel(level)
            ch = logging.StreamHandler()
            ch.setLevel(level)
            ch.setFormatter(tmp)
            logger.addHandler(ch)

        if log_file:
            self.log_file = f'{self.work_dir}/log_{self.local_rank}.txt'
            fh = logging.FileHandler(self.log_file)
            fh.setLevel(level)
            fh.setFormatter(tmp)
            logger.addHandler(fh)
        self.logger = logger
        return logger

    def make_dir(self, dir_, exist_ok):
        # We might want to limit folder creation to only local_rank 0
        os.makedirs(dir_, exist_ok=exist_ok)

    def get_tb_writer(self, tb_dir=None, exist_ok=True):
        if self.local_rank == 0:
            if tb_dir is None:
                if not hasattr(self, 'tb_dir') or self.tb_dir is None:
                    self.tb_dir = os.path.join(self.work_dir, 'tensorboard')
            else:  # if user passes in tb_dir then override the exp's tb_dir
                self.tb_dir = tb_dir
            self.make_dir(self.tb_dir, exist_ok)

            try:
                from tensorboardX import SummaryWriter
                self.tb_writer = SummaryWriter(self.tb_dir)
            except ImportError:
                self.tb_writer = None
                self.logger.info('Not using TensorBoard.')
                self.logger.info('Install tensorboardX to use TensorBoard')
        return self.tb_writer

    def log_exp_info(self, params, print_everywhere=False):
        if print_everywhere or self.local_rank == 0:
            self.logger.info("NEMO MODEL'S PARAMETERS")
            for key in params:
                self.logger.info(f'{key}\t{params[key]}')
            self.logger.info(f'Experiment output is stored in {self.work_dir}')
