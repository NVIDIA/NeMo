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

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoConfig, AutoProcessor
from nemo.collections.llm import fn
from nemo.lightning import io
from nemo.utils import logging


def masked_cross_entropy(logits, targets, mask=None):
    if mask is not None:
        loss = F.cross_entropy(logits, targets, reduction='none')
        return torch.mean(loss[mask == 1])
    else:
        return F.cross_entropy(logits, targets)


class HfAutoModelForImageTextToText(pl.LightningModule, io.IOMixin, fn.FNMixin):
    def __init__(
        self,
        model_name='gpt2',
        load_pretrained_weights=True,
        processor=None,
        loss_fn=masked_cross_entropy,
        model_transform=None,
        trust_remote_code=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self._processor = processor
        self.model = None
        self.loss_fn = loss_fn
        self.load_pretrained_weights = load_pretrained_weights
        self.is_hf_model = True
        self.model_transform = model_transform
        self.trust_remote_code = trust_remote_code

    @property
    def processor(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code
            )
        return self._processor

    @processor.setter
    def processor(self, value):
        assert self._processor is None
        self._processor = value

    @staticmethod
    def configure_processor(model_name):
        return AutoProcessor.from_pretrained(model_name)

    def configure_model(self):
        # create all your layers here
        if self.load_pretrained_weights:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name, torch_dtype='auto', trust_remote_code=self.trust_remote_code
            )
        else:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
            self.model = AutoModelForImageTextToText.from_config(config, trust_remote_code=self.trust_remote_code)
        self.model.train()

    def forward(self, batch):
        inputs = self.processor(**batch, return_tensors='pt').to(self.model.device, self.model.dtype)
        outputs = self.model(
            **inputs
        )
        labels = batch['labels'].to(self.model.device)
        if batch.get('loss_mask', None) is not None:
            loss_mask = loss_mask.to(self.model.device).view(-1)
        n_cls = outputs.logits.shape[-1]
        outputs.loss = self.loss_fn(outputs.logits.view(-1, n_cls), labels.view(-1), loss_mask)
        return outputs

    def training_step(self, batch):
        output = self.forward(batch)

        loss = output.loss
        self.log('train_log', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)

        loss = output.loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

    def save_pretrained(self, path):
        assert self.model is not None, "Model has to be created first."
        self.model.save_pretrained(path)
        if self._processor is not None:
            self._processor.save_pretrained(path)
        else:
            logging.warning("A processor wasn't created before to save.")
