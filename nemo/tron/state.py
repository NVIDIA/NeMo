# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import time
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import yaml
from megatron.core.timers import Timers
from megatron.core.utils import StragglerDetector
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.tensorboard.writer import SummaryWriter

from nemo.tron.config import ConfigContainer
from nemo.tron.tokenizers.tokenizer import build_tokenizer
from nemo.tron.utils.common_utils import dump_dataclass_to_yaml, get_rank_safe, get_world_size_safe
from nemo.tron.utils.sig_utils import DistributedSignalHandler


@dataclass
class TrainState(Stateful):
    """Dataclass to hold the state of the training process.

    Inherits from Stateful for distributed checkpointing compatibility.
    Tracks iteration count, consumed samples, flags for train/valid/test phases,
    and floating-point operations.
    """

    step: int = 0
    consumed_train_samples: int = 0
    skipped_train_samples: int = 0
    consumed_valid_samples: int = 0
    variable_seq_lengths: bool = False
    floating_point_operations_so_far: int = 0
    do_train: bool = False
    do_valid: bool = False
    do_test: bool = False

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Serializes the training state into a dictionary of tensors.

        Conforms to the Stateful interface for distributed checkpointing.

        Returns:
            A dictionary where keys are state variable names and values are
            their corresponding tensor representations.
        """
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "consumed_train_samples": torch.tensor(self.consumed_train_samples, dtype=torch.int32),
            "skipped_train_samples": torch.tensor(self.skipped_train_samples, dtype=torch.int32),
            "consumed_valid_samples": torch.tensor(self.consumed_valid_samples, dtype=torch.int32),
            "variable_seq_lengths": torch.tensor(self.variable_seq_lengths, dtype=torch.bool),
            "floating_point_operations_so_far": torch.tensor(
                self.floating_point_operations_so_far, dtype=torch.float64
            ),
            "do_train": torch.tensor(self.do_train, dtype=torch.bool),
            "do_valid": torch.tensor(self.do_valid, dtype=torch.bool),
            "do_test": torch.tensor(self.do_test, dtype=torch.bool),
        }

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load the training state from a state dictionary.

        Args:
            state_dict: A dictionary containing the state variables as tensors.
        """
        self.step = state_dict["step"].item()
        self.consumed_train_samples = state_dict["consumed_train_samples"].item()
        self.skipped_train_samples = state_dict["skipped_train_samples"].item()
        self.consumed_valid_samples = state_dict["consumed_valid_samples"].item()
        self.variable_seq_lengths = state_dict["variable_seq_lengths"].item()
        self.floating_point_operations_so_far = state_dict["floating_point_operations_so_far"].item()
        self.do_train = state_dict["do_train"].item()
        self.do_valid = state_dict["do_valid"].item()
        self.do_test = state_dict["do_test"].item()


@dataclass
class FaultToleranceState:
    """Dataclass to hold state specific to fault tolerance mechanisms."""

    ft_state_path: Optional[str] = None
    is_persistent_chkpt_loaded: bool = False
    is_async_chkpt_enabled: bool = False
    is_calculating_timeouts: bool = False
    is_setup_section_open: bool = False
    seen_checkpoints_cnt: int = 0
    seen_tr_iters_cnt: int = 0
    curr_eval_iter_idx: int = 0


# replacement for Megatron's global variables, except mbs calc and parallel state
class GlobalState:
    """Manages the global state of the training process.

    Provides access to configuration, tokenizer, loggers, timers,
    training state, fault tolerance state, signal handler, and straggler detector
    through properties with lazy initialization.
    """

    def __init__(self) -> None:
        """Initializes the GlobalState object."""
        # Prevent reinitialization in subsequent instantiations.
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self._cfg: Optional[ConfigContainer] = None
        self._tokenizer: Optional[Any] = None
        self._tensorboard_logger: Optional[SummaryWriter] = None
        self._wandb_logger: Optional[Any] = None
        self._timers: Optional[Timers] = None
        self._train_state: Optional[TrainState] = None
        self.rank_monitor_client: Optional[Any] = None
        self._signal_handler: Optional[DistributedSignalHandler] = None
        self.start_time: float = time.time()
        self._ft_state: Optional[FaultToleranceState] = None
        self._straggler_timer: Optional[StragglerDetector] = None

    @property
    def cfg(self) -> Optional[ConfigContainer]:
        """The main configuration container object."""
        return self._cfg

    @cfg.setter
    def cfg(self, value: Optional[ConfigContainer]) -> None:
        """Sets the configuration container and initializes the signal handler.

        Args:
            value: The ConfigContainer instance to set.
        """
        self._cfg = value

        # This lazily initializes the signal handler when the config is set
        # in order to read the exit signal from the config.
        # This assumes the global state is first initialized and that the
        # config is immediately set on the global state after initialization.
        if value is not None:
            self._set_signal_handler()

    @property
    def tokenizer(self) -> Any:
        """The tokenizer instance, lazily built based on the config."""
        if self._tokenizer is None:
            self._tokenizer = build_tokenizer(self.cfg.tokenizer_config)
        return self._tokenizer

    @property
    def tensorboard_logger(self) -> Optional[SummaryWriter]:
        """The TensorBoard SummaryWriter instance, lazily initialized for rank N-1."""
        if self._tensorboard_logger is None:
            if self.cfg.logger_config.tensorboard_dir and get_rank_safe() == (get_world_size_safe() - 1):
                from torch.utils.tensorboard.writer import SummaryWriter

                print("> setting tensorboard ...")
                self._tensorboard_logger = SummaryWriter(
                    log_dir=self.cfg.logger_config.tensorboard_dir,
                    max_queue=self.cfg.logger_config.tensorboard_queue_size,
                )
            else:
                self._tensorboard_logger = None
        return self._tensorboard_logger

    @property
    def wandb_logger(self) -> Optional[Any]:
        """The Weights & Biases logger instance, lazily initialized for rank N-1."""
        if self._wandb_logger is None:
            if self.cfg.logger_config.wandb_project and get_rank_safe() == (get_world_size_safe() - 1):
                if self.cfg.logger_config.wandb_exp_name == "":
                    raise ValueError("Please specify the wandb experiment name!")

                import wandb

                save_dir = self.cfg.logger_config.wandb_save_dir or os.path.join(self.cfg.save, "wandb")
                wandb_kwargs = {
                    "dir": save_dir,
                    "name": self.cfg.logger_config.wandb_exp_name,
                    "project": self.cfg.logger_config.wandb_project,
                    "config": yaml.safe_load(dump_dataclass_to_yaml(self.cfg)),
                }
                os.makedirs(wandb_kwargs["dir"], exist_ok=True)
                wandb.init(**wandb_kwargs)

                self._wandb_logger = wandb
            else:
                self._wandb_logger = None
        return self._wandb_logger

    @property
    def timers(self) -> Timers:
        """The Megatron Timers instance used for tracking execution times."""
        if self._timers is None:
            self._timers = Timers(self.cfg.logger_config.timing_log_level, self.cfg.logger_config.timing_log_option)
            self._timers.write_to_wandb = types.MethodType(_timers_write_to_wandb, self._timers)
        return self._timers

    @property
    def train_state(self) -> TrainState:
        """The TrainState object holding training progress information."""
        if self._train_state is None:
            self._train_state = TrainState()
        return self._train_state

    @train_state.setter
    def train_state(self, value: TrainState) -> None:
        """Sets the training state object.

        Args:
            value: The TrainState instance to set.
        """
        self._train_state = value

    @property
    def fault_tolerance_state(self) -> FaultToleranceState:
        """The FaultToleranceState object holding FT-specific information."""
        if self._ft_state is None:
            self._ft_state = FaultToleranceState()
        return self._ft_state

    @fault_tolerance_state.setter
    def fault_tolerance_state(self, value: FaultToleranceState) -> None:
        """Sets the fault tolerance state object.

        Args:
            value: The FaultToleranceState instance to set.
        """
        self._ft_state = value

    @property
    def signal_handler(self) -> DistributedSignalHandler:
        """The DistributedSignalHandler instance for graceful shutdown."""
        if self._signal_handler is None:
            self._set_signal_handler()
        return self._signal_handler

    @property
    def straggler_timer(self) -> StragglerDetector:
        """The StragglerDetector instance for tracking slow GPUs."""
        if self._straggler_timer is None:
            self._straggler_timer = StragglerDetector()
        return self._straggler_timer

    def _set_signal_handler(self) -> None:
        """Initializes the distributed signal handler based on the configuration."""
        self._signal_handler = DistributedSignalHandler(self.cfg.train_config.exit_signal)


def _timers_write_to_wandb(
    self: Timers,
    names: List[str],
    writer: Any,
    iteration: int,
    normalizer: float = 1.0,
    reset: bool = True,
    barrier: bool = False,
) -> None:
    """Patch to write timers to wandb for Megatron Core Timers."""
    # currently when using add_scalars,
    # torch.utils.add_scalars makes each timer its own run, which
    # polutes the runs list, so we just add each as a scalar
    assert normalizer > 0.0
    name_to_min_max_time = self._get_global_min_max_time(names, reset, barrier, normalizer)
    if writer is not None:
        for name in name_to_min_max_time:
            _, max_time = name_to_min_max_time[name]
            writer.log({name + "-time": max_time}, iteration)
