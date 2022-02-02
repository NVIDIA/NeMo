# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import Any, Dict, Optional, Union

import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.glue_benchmark.glue_benchmark_dataset import TextToTextGLUEDataset, TextToTextXNliDataset
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.utils import logging
from nemo.collections.nlp.models.language_modeling.megatron_glue_model import MegatronT5GLUEModel
from collections import Counter

try:
    from apex.transformer import tensor_parallel
    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__ = ['MegatronXNliModel']


class MegatronXNliModel(MegatronT5GLUEModel):

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg=cfg, trainer=trainer)
        self.cfg = cfg

    def process_batch(self, batch):
        """Build the batch."""

        keys = ['text_enc', 'text_dec', 'labels', 'loss_mask', 'enc_mask', 'dec_mask', 'enc_dec_mask']
        datatype = torch.int64

        data = batch
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_enc = data_b['text_enc'].long()
        tokens_dec = data_b['text_dec'].long()
        labels = data_b['labels'].long()
        loss_mask = data_b['loss_mask'].float()

        enc_mask = data_b['enc_mask'] < 0.5
        dec_mask = data_b['dec_mask'] < 0.5
        enc_dec_mask = data_b['enc_dec_mask'] < 0.5
        if 'lang' in batch:
            lang = batch['lang']
            return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask, lang
        else:
            return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask


    # def training_step(self, batch, batch_idx):
    #     tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask, _ = self.process_batch(batch)

    #     output_tensor, encoder_hidden_states = self.model(
    #         tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask, tokentype_ids=None, lm_labels=labels
    #     )

    #     loss = self.model.loss_func(loss_mask, output_tensor)
    #     self.log('train_loss', loss)
    #     # Reduced loss for logging.
    #     reduced_loss = average_losses_across_data_parallel_group([loss])
    #     # cache reduced loss while accumulating gradients
    #     self.model._reduced_loss_buffer.append(reduced_loss[0])

    #     if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
    #         # Reduced loss for logging.
    #         average_reduced_loss = sum(self.model._reduced_loss_buffer) / len(self.model._reduced_loss_buffer)
    #         self.log('reduced_train_loss', average_reduced_loss, prog_bar=True)
    #         lr = self._optimizer.param_groups[0]['lr']
    #         self.log('lr', lr)
    #         self.log('global_step', self.trainer.global_step, prog_bar=True)
    #         self.model._reduced_loss_buffer = []

    #     return loss

    def inference_step(self, batch, batch_idx):
        loss = self.model.validation_step(batch, batch_idx)

        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask, lang = self.process_batch(batch)

        predicted_token_ids, log_probs = self.model.decode(
            tokens_enc=tokens_enc, enc_mask=enc_mask, num_tokens_to_generate=10
        )

        return {'loss': loss, 'predicted_token_ids': predicted_token_ids, 'labels': labels, 'lang': lang}

    def inference_epoch_end(self, outputs):
        losses = [x['loss'] for x in outputs]
        averaged_loss = average_losses_across_data_parallel_group(losses)
        all_preds = []
        all_labels = []
        all_langs = []
        for item in outputs:
            preds = item['predicted_token_ids'].cpu().numpy().tolist()
            labels = item['labels'].cpu().numpy().tolist()
            all_langs.extend(item['lang'])
            for i, (pred, label) in enumerate(zip(preds, labels)):
                if self.model.tokenizer.eos_id in pred:
                    idx = pred.index(self.model.tokenizer.eos_id)
                    pred = pred[:idx]
                pred = [id for id in pred if id not in self.model.tokenizer.special_token_to_id.values()]
                label = [id for id in label if id not in self.model.tokenizer.special_token_to_id.values()]
                pred = self.model.tokenizer.ids_to_text(pred)
                label = self.model.tokenizer.ids_to_text(label)
                all_preds.append(pred)
                all_labels.append(label)

        lang_counter = Counter()
        lang_correct_counter = Counter()
        correct = 0
        for pred, label, lang in zip(all_preds, all_labels, all_langs):
            lang_counter.update({lang: 1})
            if pred == label:
                correct += 1
                lang_correct_counter.update({lang: 1})
        lang_acc = {}
        for key in lang_counter:
            lang_acc[key] =  lang_correct_counter[key] / lang_counter[key]
        acc = correct / len(all_preds)
        return averaged_loss[0], acc, lang_acc

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        val_loss, val_acc, lang_acc = self.inference_epoch_end(outputs)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)
        for key in lang_acc:
            self.log(f'val_{key}_acc', lang_acc[key])
        logging.info(f'Validation loss: {val_loss}')
        logging.info(f'Validation accuracy: {val_acc}')
        for key in lang_acc:
            logging.info(f'Validation {key} accuracy: {lang_acc[key]}')

    def test_epoch_end(self, outputs):
        test_loss, test_acc, lang_acc = self.inference_epoch_end(outputs)
        self.log('test_loss',test_loss, prog_bar=True)
        self.log('test_acc', test_acc, prog_bar=True)
        for key in lang_acc:
            self.log(f'test_{key}_acc', lang_acc[key])
        logging.info(f'Test loss: {test_loss}')
        logging.info(f'Test accuracy: {test_acc}')
        for key in lang_acc:
            logging.info(f'Test {key} accuracy: {lang_acc[key]}')

    def build_train_valid_test_datasets(self, test_only=False):
        logging.info('Building GLUE datasets.')
        self._test_ds = TextToTextXNliDataset(
            self.cfg.data.test_ds.file_path,
            task_name=self.cfg.data.test_ds.task_name,
            tokenizer=self.model.tokenizer,
            max_seq_length=self.cfg.data.test_ds.max_seq_length,
        )
        if test_only:
            return None, None, self._test_ds
        self._train_ds = TextToTextGLUEDataset(
            self.cfg.data.train_ds.file_path,
            task_name=self.cfg.data.train_ds.task_name,
            tokenizer=self.model.tokenizer,
            max_seq_length=self.cfg.data.train_ds.max_seq_length,
        )
        self._validation_ds = TextToTextXNliDataset(
            self.cfg.data.validation_ds.file_path,
            task_name=self.cfg.data.validation_ds.task_name,
            tokenizer=self.model.tokenizer,
            max_seq_length=self.cfg.data.validation_ds.max_seq_length,
        )
        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building T5 datasets.')
        return self._train_ds, self._validation_ds, self._test_ds
