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

import fiddle as fdl
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim

from nemo.collections.llm import fn
from nemo.lightning import io


class GenericLitWrapper(pl.LightningModule, io.IOMixin, fn.FNMixin):
    def __init__(self, model, criterion, model_transform=None):
        super(GenericLitWrapper, self).__init__()
        self.module = model
        self.criterion = criterion
        self.model_transform = model_transform
        self.tokenizer = None

    def configure_model(self):
        if isinstance(self.module, fdl.Config):
            self.module = fdl.build(self.module)

    def forward(self, x):
        if isinstance(x, dict):
            return self.module(**x)
        else:
            return self.module(x)

    def training_step(self, batch, batch_idx):
        y = batch.pop('labels')
        y_hat = self(batch)
        # Calculate loss
        logits = extract_logits_from_output(y_hat)
        n = logits.shape[-1]
        loss = self.criterion(logits.view(-1, n), y.view(-1))
        self.log('train_log', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss


def extract_logits_from_output(x):
    if isinstance(x, torch.Tensor):
        return x
    elif hasattr(x, 'logits'):
        return x.logits
    else:
        raise ValueError("Unable to extract model's logits; looked for .logits attr")


def wrap_module_with_lit(model, criterion_fn):
    if isinstance(model, nn.Module) and isinstance(model, pl.LightningModule):
        # Already a pl.LightningModule, nothing to do.
        return model
    elif isinstance(model, (fdl.Config, nn.Module)):
        # It's a fdl.Config or a nn.Module -> wrap it with a GenericLitWrapper
        return GenericLitWrapper(model, criterion_fn())
    else:
        raise ValueError("Expected model to be of type pl.LightningModule, fdl.Config or nn.Module")
