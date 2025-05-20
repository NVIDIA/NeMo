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

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.core import LightningModule


class IPLEpochStopper(Callback):
    r"""
    Gracefully terminates training at the *end* of an epoch.
    This is done to generate pseudo-labels dynamically.
    enable_stop : bool, default=False
        If ``True`` the callback will request a stop in
        :py:meth:`on_train_epoch_end`.  If ``False`` it is inert.
    """

    def __init__(self, enable_stop: bool = False) -> None:
        super().__init__()
        self.enable_stop = bool(enable_stop)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Sets `should_stop` stop flag to terminate the training.
        """
        super().__init__()
        if self.enable_stop:
            trainer.should_stop = True
