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


from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.trainer import Trainer

from nemo.lightning.io.mixin import IOMixin
from nemo.lightning.pytorch.accelerate.transformer_engine import accelerate, apply_fp8_autocast


def extract_module_attr_name(pl_module: "pl.LightningModule") -> str:
    if hasattr(pl_module, 'module'):
        return 'module'
    elif hasattr(pl_module, 'model'):
        return 'model'
    else:
        raise ValueError("Expected lightning_module to have a .model or .module attr.")


class TETransform(Callback, IOMixin):
    """
    Apply TEAccelerator on HF model
    Args:
        fp8_autocast (bool): Applies TE's fp8 autocast if true
    Example:
        >>> from nemo.lightning.pytorch.callbacks import TETransform
        >>> trainer = Trainer(callbacks=[TETransform()])
    """

    def __init__(self, fp8_autocast=False):
        self.fp8_autocast = fp8_autocast

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        attr_name = extract_module_attr_name(pl_module)
        model = getattr(pl_module, attr_name)

        accelerate(model)
        if self.fp8_autocast:
            apply_fp8_autocast(model)
