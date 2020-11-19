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

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import MISSING

from nemo.core import config
from nemo.core.classes.dataset import DatasetConfig
from nemo.utils import exp_manager


@dataclass
class SchedConfig:
    name: str = MISSING
    min_lr: float = 0.0
    last_epoch: int = -1


@dataclass
class OptimConfig:
    name: str = MISSING
    lr: float = MISSING
    sched: Optional[SchedConfig] = None


@dataclass
class ModelConfig:
    """
    Model component inside ModelPT
    """

    # ...
    train_ds: Optional[DatasetConfig] = None
    validation_ds: Optional[DatasetConfig] = None
    test_ds: Optional[DatasetConfig] = None
    optim: Optional[OptimConfig] = None


@dataclass
class HydraConfig:
    run: Dict[str, Any] = field(default_factory=lambda: {"dir": "."})
    job_logging: Dict[str, Any] = field(default_factory=lambda: {"root": {"handlers": None}})


@dataclass
class ModelPTConfig:
    name: str = MISSING
    model: ModelConfig = MISSING
    trainer: config.TrainerConfig = config.TrainerConfig(
        accelerator="ddp", checkpoint_callback=False, logger=False, log_every_n_steps=1
    )
    exp_manager: Optional[Any] = exp_manager.ExpManagerConfig()
    hydra: HydraConfig = HydraConfig()
