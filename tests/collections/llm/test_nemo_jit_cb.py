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

from typing import Optional

import fiddle as fdl
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm import fn
from nemo.lightning import io
from nemo.lightning.io.mixin import track_io
from nemo.lightning.pytorch.callbacks import JitTransform


class SquadDataModuleWithPthDataloader(llm.SquadDataModule):
    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=dataset.collate_fn,
            batch_size=self.micro_batch_size,
            **kwargs,
        )


def squad(tokenizer) -> pl.LightningDataModule:
    return SquadDataModuleWithPthDataloader(
        tokenizer=tokenizer,
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=1,  # assert gbs == mbs * accumulate_grad_batches
        num_workers=0,
        dataset_kwargs={"sanity_check_dist_workers": False},
    )


@track_io
class OrdTokenizer:
    def __init__(self, vocab_size=30_000, num_reserved_tokens=128, special_token_names=['bos_id', 'eos_id', 'pad_id']):
        self.vocab_size = vocab_size
        self.num_reserved_tokens = num_reserved_tokens
        self.special_token_names = special_token_names
        assert len(self.special_token_names) < num_reserved_tokens

    def __getattr__(self, name):
        if name in self.__dict__.get('special_token_names', {}):
            return self.__dict__['special_token_names'].index(name)
        elif name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError

    def text_to_ids(self, text):
        token_ids = list(map(lambda x: self.num_reserved_tokens + ord(x), list(text)))
        assert max(token_ids) < self.vocab_size
        return token_ids


class JitTestModel(pl.LightningModule, io.IOMixin, fn.FNMixin):
    def __init__(
        self,
        tokenizer=None,
        has_jit=False,
    ):
        super().__init__()
        self.has_jit = has_jit
        self.tokenizer = tokenizer

    def configure_model(self) -> None:
        if not hasattr(self, "module"):
            self.module = nn.Sequential(
                nn.Embedding(30_000, 512),
                nn.TransformerEncoderLayer(512, 8, 4096, dropout=0.1),
                nn.Linear(512, 30_000),
            )

    def forward(self, input_ids, attention_mask=None, labels=None, loss_mask=None):
        output = self.module(input_ids)
        expected_cls = (
            torch._dynamo.eval_frame.OptimizedModule if self.has_jit else torch.nn.modules.container.Sequential
        )
        assert isinstance(self.module, expected_cls), type(self.module)
        return F.cross_entropy(output.view(-1, output.shape[-1]), labels.view(-1))

    def training_step(self, batch):
        tokens = batch['tokens']
        labels = batch['labels']
        loss_mask = batch.get('loss_mask', None)
        return self.forward(
            input_ids=tokens,
            labels=labels,
            loss_mask=loss_mask,
        )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', default=1)
    parser.add_argument('--max-steps', type=int, default=1)
    args = parser.parse_args()

    for has_jit in [True, False]:
        tokenizer = OrdTokenizer()
        model = JitTestModel(tokenizer=tokenizer, has_jit=has_jit)
        optim = fdl.build(llm.sgd.pytorch_sgd_with_flat_lr(max_lr=1e-5))

        llm.api.finetune(
            model=model,
            data=squad(tokenizer),
            trainer=nl.Trainer(
                devices=args.devices,
                max_steps=args.max_steps,
                accelerator='gpu',
                strategy='auto',
                log_every_n_steps=1,
                limit_val_batches=0.0,
                num_sanity_val_steps=0,
                accumulate_grad_batches=1,
                gradient_clip_val=1.0,
                use_distributed_sampler=False,
                callbacks=[JitTransform('torch' if has_jit else None)],
            ),
            optim=optim,
            log=None,
        )
