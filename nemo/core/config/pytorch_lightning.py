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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = ['TrainerConfig']


@dataclass
class TrainerConfig:
    """
    Configuration of PyTorch Lightning Trainer.

    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).

    ..warning:
        Picked just few params of the PTL trainer for now. This needs to be discussed.

    ..note:
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#
    """

    gradient_clip_val: float = 0
    process_position: int = 0
    num_nodes: int = 1
    num_processes: int = 1
    gpus: Optional[int] = None
    auto_select_gpus: bool = False
    log_gpu_memory: Optional[str] = None
    progress_bar_refresh_rate: int = 1
    check_val_every_n_epoch: int = 1
    fast_dev_run: bool = False
    max_epochs: int = 1000
    min_epochs: int = 1
    distributed_backend: Optional[str] = None
    max_steps: Optional[int] = None
    accumulate_grad_batches: int = 1
    amp_level: str = "O0"
