import os

from megatron.core import Timers

from nemo.tron.config import FlatConfig
from nemo.tron.init import get_rank_safe, get_world_size_safe


# replacement for Megatron's global variables, except mbs calc and parallel state
class ObjectStore:
    """Stores loggers, timers, and tokenizer."""

    def __init__(self, cfg: FlatConfig, build_tokenizer=True):
        if build_tokenizer:
            self._init_tokenizer(cfg)
        self._init_tb_logger(cfg)
        self._init_wandb_logger(cfg)
        self._init_timers(cfg)

        # if cfg.exit_signal_handler: TODO (maanug): do we need this?
        #     self._set_signal_handler()

    def _init_tokenizer(self, cfg: FlatConfig):
        # TODO (maanug): implement. use collections.common.tokenizers, result may look different than MLM
        pass

    def _init_tb_logger(self, cfg: FlatConfig):
        self.tensorboard_logger = None

        if hasattr(cfg, 'tensorboard_dir') and cfg.tensorboard_dir and get_rank_safe() == (get_world_size_safe() - 1):
            from torch.utils.tensorboard.writer import SummaryWriter

            print('> setting tensorboard ...')
            self.tensorboard_logger = SummaryWriter(log_dir=cfg.tensorboard_dir, max_queue=cfg.tensorboard_queue_size)

    def _init_wandb_logger(self, cfg: FlatConfig):
        self.wandb_logger = None

        if getattr(cfg, 'wandb_project', '') and get_rank_safe() == (get_world_size_safe() - 1):
            if cfg.wandb_exp_name == '':
                raise ValueError("Please specify the wandb experiment name!")

            import wandb

            if cfg.wandb_save_dir:
                save_dir = cfg.wandb_save_dir
            else:
                # Defaults to the save dir.
                save_dir = os.path.join(cfg.save, 'wandb')
            wandb_kwargs = {
                'dir': save_dir,
                'name': cfg.wandb_exp_name,
                'project': cfg.wandb_project,
                'config': vars(cfg),
            }
            os.makedirs(wandb_kwargs['dir'], exist_ok=True)
            wandb.init(**wandb_kwargs)

            self.wandb_logger = wandb

    def _init_timers(self, cfg: FlatConfig):
        self.timers = Timers(cfg.timing_log_level, cfg.timing_log_option)
