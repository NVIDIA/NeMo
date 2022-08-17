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
import torch

from copy import deepcopy
from typing import List, Dict, Any, Optional

try:
    import amp_C

    apex_available = True
except Exception:
    apex_available = False

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT


class EMA(Callback):
    """
    Implements Exponential Moving Averaging (EMA).

    When training a model, this callback will maintain moving averages of the trained parameters.
    When evaluating, we use the moving averages copy of the trained parameters.

    Args:
        ema: The exponential decay used when calculating the moving average. Has to be between 0-1.
        apply_ema_every_n_steps: Apply EMA every n global steps.
        start_step: Start applying EMA from ``start_step`` global step onwards.
    """

    def __init__(self, ema: float, apply_ema_every_n_steps: int = 1, start_step: int = 0):
        if not apex_available:
            raise MisconfigurationException(
                "EMA requires Apex to be installed: https://github.com/NVIDIA/apex#installation."
            )
        if not (0 <= ema <= 1):
            raise MisconfigurationException("EMA decay value must be between 0 and 1")
        self._ema_model: Optional["pl.LightningModule"] = None
        self._ema_model_weights: Optional[List[torch.Tensor]] = None
        self._overflow_buf: Optional[torch.Tensor] = None
        self._cur_step: Optional[int] = None
        self._weights_buffer: Optional[List[torch.Tensor]] = None
        self.apply_ema_every_n_steps = apply_ema_every_n_steps
        self.start_step = start_step
        self.ema = ema

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if pl_module.device.type != "cuda":
            raise MisconfigurationException("Apex EMA Callback only works with CUDA. Ensure to set accelerator='gpu'.")
        with pl_module._prevent_trainer_and_dataloaders_deepcopy():
            self._ema_model = deepcopy(pl_module)
        self.init_multi_tensor_ema(pl_module, self._ema_model)

    def init_multi_tensor_ema(self, model: "pl.LightningModule", ema_model: "pl.LightningModule") -> None:
        self._ema_model_weights = list(ema_model.state_dict().values())
        self._overflow_buf = torch.IntTensor([0]).to(model.device)

    def apply_multi_tensor_ema(self, pl_module: "pl.LightningModule") -> None:
        model_weights = list(pl_module.state_dict().values())
        amp_C.multi_tensor_axpby(
            65536, # todo (sean): chunk size, should we expose?
            self._overflow_buf,
            [self._ema_model_weights, model_weights, self._ema_model_weights],
            self.ema,
            1 - self.ema,
            -1,
        )

    def should_apply_ema(self, step: int) -> bool:
        return step != self._cur_step and step >= self.start_step and step % self.apply_ema_every_n_steps == 0

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if self.should_apply_ema(trainer.global_step):
            self._cur_step = trainer.global_step
            self.apply_multi_tensor_ema(pl_module)

    def state_dict(self) -> Dict[str, Any]:
        return dict(ema_weights=self._ema_model_weights, cur_step=self._cur_step)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._ema_model_weights = state_dict['ema_weights']
        self._cur_step = state_dict['cur_step']

    def replace_model_weights(self, pl_module: "pl.LightningModule") -> None:
        self._weights_buffer = [p.detach().to('cpu').clone() for p in pl_module.state_dict().values()]
        new_state_dict = {k: v for k, v in zip(pl_module.state_dict().keys(), self._ema_model_weights)}
        pl_module.load_state_dict(new_state_dict)

    def restore_model_weights(self, pl_module: "pl.LightningModule") -> None:
        state_dict = pl_module.state_dict()
        new_state_dict = {k: v for k, v in zip(state_dict.keys(), self._weights_buffer)}
        pl_module.load_state_dict(new_state_dict)

    @property
    def ema_initialized(self) -> bool:
        return self._ema_model_weights is not None

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized:
            self.replace_model_weights(pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized:
            self.restore_model_weights(pl_module)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized:
            self.replace_model_weights(pl_module)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized:
            self.restore_model_weights(pl_module)
