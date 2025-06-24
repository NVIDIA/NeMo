# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Dict, List, Tuple, Union
import math

import pytorch_lightning as pl
from lightning.pytorch.callbacks.callback import Callback

from nemo.lightning.io.mixin import IOMixin


class LayerFreezer(Callback, IOMixin):
    """
    Freezes sub-modules of a LightningModule based on the list provided. The list of layers should
    be the full FQN.

    Instantiate
    -----------
    callback = LayerFreezer(['layer1', 'layer2',])
    trainer  = pl.Trainer(callbacks=[callback], ...)
    """

    def __init__(self, frozen_layers: List[str]):
        """
        Args
        ----
        frozen_layers: List[str] list of layers that are frozen
        """
        super().__init__()
        self.frozen_layers = frozen_layers

    @staticmethod
    def _resolve_attr(root, path: str):
        """
        Traverse dotted attribute path (“encoder.layer1”) from root.
        """
        m = root
        for part in path.split('.'):
            m = getattr(m, part)
        return m

    def _apply_freeze(self, module, freeze: bool):
        """
        Enable/disable gradients + switch (eval/train) mode.
        """
        for p in module.parameters():
            p.requires_grad = not freeze
        # Optional: also flip training mode so dropout / BN are disabled.
        module.eval() if freeze else module.train()

    def on_train_batch_start(self, trainer, pl_module, *_):
        for name in self.frozen_layers:
            submod = self._resolve_attr(pl_module, name)
            self._apply_freeze(submod, should_be_frozen)
            self.frozen_state[name] = should_be_frozen

    # In case we resume from checkpoint, re-establish correct state
    def on_train_start(self, trainer, pl_module):
        self.on_train_batch_start(trainer, pl_module, None, 0)
