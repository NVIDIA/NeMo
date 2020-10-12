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
import time

import numpy as np
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_only

from nemo.collections.common.metrics.sacrebleu import corpus_bleu


class LogEpochTimeCallback(Callback):
    """Simple callback that logs how long each epoch takes, in seconds, to a pytorch lightning log
    """

    @rank_zero_only
    def on_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()

    @rank_zero_only
    def on_epoch_end(self, trainer, pl_module):
        curr_time = time.time()
        duration = curr_time - self.epoch_start
        trainer.logger.log_metrics({"epoch_time": duration}, step=trainer.global_step)

class MachineTranslationLogEvalCallback(Callback):
    def _on_eval_end(self, trainer, pl_module, mode):
        counts = np.array(self._non_pad_tokens)
        eval_loss = np.sum(np.array(self._losses) * counts) / np.sum(counts)
        token_bleu = corpus_bleu(self._translations, [self._ground_truths], tokenize="fairseq")
        sacre_bleu = corpus_bleu(self._translations, [self._ground_truths], tokenize="13a")
        print(f"{mode} results".capitalize())
        for i in range(3):
            sent_id = np.random.randint(len(self._translations))
            print(f"Ground truth: {self._ground_truths[sent_id]}\n")
            print(f"Translation: {self._translations[sent_id]}\n")
        print("-" * 50)
        print(f"loss: {eval_loss:.3f}")
        print(f"TokenBLEU: {token_bleu}")
        print(f"SacreBLEU: {sacre_bleu}")
        print("-" * 50)

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        self._on_eval_end(trainer, pl_module, "Test")

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        self._on_eval_end(trainer, pl_module, "Validation")

    @rank_zero_only
    def on_sanity_check_end(self, trainer, pl_module):
        self._on_eval_end(trainer, pl_module, "Validation")

    def _on_eval_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        for tr in pl_module.last_eval_beam_results.cpu().numpy():
            self._translations.append(pl_module.tgt_tokenizer.ids_to_text(tr))
        tgts = batch[2].squeeze(dim=0).cpu().numpy()
        for tgt in tgts:
            self._ground_truths.append(pl_module.tgt_tokenizer.ids_to_text(tgt))
        non_pad_tokens = np.not_equal(tgts, pl_module.tgt_tokenizer.pad_id).sum().item()
        self._non_pad_tokens.append(non_pad_tokens)
        self._losses.append(pl_module.last_eval_loss)

    @rank_zero_only
    def on_test_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._on_eval_batch_end(trainer, pl_module, batch, batch_idx, dataloader_idx)

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._on_eval_batch_end(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def _on_eval_start(self, trainer, pl_module):
        self._translations = []
        self._ground_truths = []
        self._losses = []
        self._non_pad_tokens = []

    @rank_zero_only
    def on_test_start(self, trainer, pl_module):
        self._on_eval_start(trainer, pl_module)

    @rank_zero_only
    def on_validation_start(self, trainer, pl_module):
        self._on_eval_start(trainer, pl_module)

    @rank_zero_only
    def on_sanity_check_start(self, trainer, pl_module):
        self._on_eval_start(trainer, pl_module)

