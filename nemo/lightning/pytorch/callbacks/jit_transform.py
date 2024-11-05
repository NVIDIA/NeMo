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

from typing import Optional

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.trainer import Trainer

from nemo.lightning.io.mixin import IOMixin


def extract_module_attr_name(pl_module: "pl.LightningModule") -> str:
    if hasattr(pl_module, 'module'):
        return 'module'
    elif hasattr(pl_module, 'model'):
        return 'model'
    else:
        raise ValueError("Expected lightning_module to have a .model or .module attr.")


class JitTransform(Callback, IOMixin):
    """
    Apply JIT-compling on PyTorch model

    Args:
        backend (str, optional): The jit-compiler backend to use.

    Example:
        >>> from nemo.lightning.pytorch.callbacks import JitTransform
        >>> trainer = Trainer(callbacks=[JitTransform('torch)])
    """

    def __init__(self, backend: Optional[int] = None):
        if backend is not None:
            assert isinstance(backend, str)
            backend = backend.lower()
            assert backend in ['torch', 'thunder']
        self.backend = backend

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.backend is None:
            return
        attr_name = extract_module_attr_name(pl_module)
        model = getattr(pl_module, attr_name)
        if self.backend == 'torch':
            jit_model = torch.compile(model)
        elif self.backend == 'thunder':
            import thunder

            jit_model = thunder.jit(model)
        else:
            raise ValueError("got unexpected backend")
        setattr(trainer.lightning_module, attr_name, jit_model)
