# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Type

import torch

from nemo.automodel.llm.causal_lm import AutoModelForCausalLMConfig
from nemo.automodel.loss.linear_ce import HAVE_LINEAR_LOSS_CE
from nemo.tron.config import (
    DistributedInitConfig,
    FinetuningDatasetConfig,
    LoggerConfig,
    ProfilingConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from nemo.tron.utils.common_utils import get_world_size_safe
from nemo.tron.utils.config_utils import ConfigContainer as Container

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class OptimizerConfig:
    """
    Configuration for the optimizer.
    """

    optimizer_cls: Type[torch.optim.Optimizer] | Literal["te_adam"] = torch.optim.AdamW
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)

    lr: Optional[float] = None
    """Initial learning rate. Depending on decay style and initial warmup, the learning rate at each
       iteration would be different.
    """

    min_lr: Optional[float] = None
    """Minumum value for learning rate. The scheduler clip values below this threshold."""

    weight_decay: Optional[float] = None
    """Weight decay for the optimizer."""

    barrier_with_L1_time: bool = False
    """Whether to use a barrier with L1 timer."""

    clip_grad: float = 1.0
    """Gradient clipping based on global L2 norm."""

    def __post_init__(self):
        if isinstance(self.optimizer_cls, str):
            if self.optimizer_cls == "te_adam":
                from transformer_engine.pytorch.optimizers import FusedAdam as Adam

                self.optimizer_cls = Adam
            else:
                raise ValueError(
                    f"Invalid string for optimizer class: {self.optimizer_cls}. " "Must be one of: 'te_adam'"
                )


@dataclass(kw_only=True)
class CheckpointConfig:
    # ---------------- Checkpointing config. ----------------

    save: Optional[str] = None
    """Output directory to save checkpoints to."""

    save_interval: Optional[int] = None
    """Number of iterations between persistent checkpoint saves."""

    save_optim: bool = True
    """Do not save current optimizer."""

    save_rng: bool = True
    """Do not save current rng state."""

    load: Optional[str] = None
    """Directory containing a model checkpoint."""

    load_optim: bool = True
    """Do not load optimizer when loading checkpoint."""

    load_rng: bool = True
    """Do not load rng state when loading checkpoint."""

    pretrained_checkpoint: Optional[str] = None
    """Directory containing a pretrained model checkpoint for finetuning."""


@dataclass(kw_only=True)
class ConfigContainer(Container):
    model_config: AutoModelForCausalLMConfig
    train_config: TrainingConfig
    optimizer_config: OptimizerConfig
    scheduler_config: SchedulerConfig
    dataset_config: FinetuningDatasetConfig
    logger_config: LoggerConfig
    tokenizer_config: TokenizerConfig
    checkpoint_config: CheckpointConfig
    dist_config: DistributedInitConfig = field(default_factory=DistributedInitConfig)
    rng_config: RNGConfig = field(default_factory=RNGConfig)
    profiling_config: Optional[ProfilingConfig] = None

    def validate(self):
        # Distributed
        world_size = get_world_size_safe()

        # TODO: Add Model Parallel support
        total_model_size = 1
        assert (
            world_size % total_model_size == 0
        ), f"world size ({world_size}) is not divisible by total_model_size ({total_model_size})"
        self.data_parallel_size = world_size // total_model_size

        if self.dist_config.lazy_init:
            # Use CPU for model initialization in lazy mode
            self.model_config.device_map = "cpu"

        # Scheduler
        if self.scheduler_config.lr_decay_iters is None:
            self.scheduler_config.lr_decay_iters = self.train_config.train_iters
        self.scheduler_config.lr_decay_steps = (
            self.scheduler_config.lr_decay_iters * self.train_config.global_batch_size
        )
        self.scheduler_config.wd_incr_steps = self.train_config.train_iters * self.train_config.global_batch_size
        self.scheduler_config.wsd_decay_steps = None
        if self.scheduler_config.lr_wsd_decay_iters is not None:
            self.scheduler_config.wsd_decay_steps = (
                self.scheduler_config.lr_wsd_decay_iters * self.train_config.global_batch_size
            )
        if self.scheduler_config.lr_warmup_fraction is not None:
            self.scheduler_config.lr_warmup_steps = (
                self.scheduler_config.lr_warmup_fraction * self.scheduler_config.lr_decay_iters
            )
        else:
            self.scheduler_config.lr_warmup_steps = (
                self.scheduler_config.lr_warmup_iters * self.train_config.global_batch_size
            )
