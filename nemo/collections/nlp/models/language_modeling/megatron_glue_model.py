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

from nemo.collections.nlp.data.glue_benchmark.glue_benchmark_dataset import TextToTextGLUEDataset
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.utils import logging

__all__ = ['MegatronT5FineTuneModel']


class MegatronT5FineTuneModel(NLPModel):
    """
    Megatron T5 finetuning
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        self.model = MegatronT5Model.restore_from(cfg.restore_from_path, trainer=trainer)

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def process_batch(self, batch):
        return self.model.process_batch(batch)

    def build_train_valid_test_datasets(self):
        pass

    def build_pretraining_data_loader(self, dataset):
        pass

    def setup(self, stage=None):
        pass

    def setup_training_data(self):
        pass

    def setup_validation_data(self):
        pass

    def setup_test_data(self):
        pass

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        """PTL hook that is called after unscaling gradients when using native amp.
           We use gradient clipping implementation from megatron-lm.
        """
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return self.model.prediction_step(batch, batch_idx, dataloader_idx)


class MegatronT5GLUEModel(MegatronT5FineTuneModel):
    """
    Megatron T5 finetuning for GLUE
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg=cfg, trainer=trainer)
        self.cfg = cfg

    def training_step(self, batch, batch_idx):
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask = self.process_batch(batch)

        output_tensor, encoder_hidden_states = self.model(
            tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask, tokentype_ids=None, lm_labels=labels
        )

        loss = self.model.loss_func(loss_mask, output_tensor)
        self.log('train_loss', loss)
        # Reduced loss for logging.
        reduced_loss = average_losses_across_data_parallel_group([loss])
        # cache reduced loss while accumulating gradients
        self.model._reduced_loss_buffer.append(reduced_loss[0])

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            # Reduced loss for logging.
            average_reduced_loss = sum(self.model._reduced_loss_buffer) / len(self.model._reduced_loss_buffer)
            self.log('reduced_train_loss', average_reduced_loss, prog_bar=True)
            lr = self._optimizer.param_groups[0]['lr']
            self.log('lr', lr)
            self.log('global_step', self.trainer.global_step, prog_bar=True)
            self.model._reduced_loss_buffer = []

        return loss

    def inference_step(self, batch, batch_idx):
        loss = self.model.validation_step(batch, batch_idx)

        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask = self.process_batch(batch)

        predicted_token_ids, log_probs = self.model.decode(
            tokens_enc=tokens_enc, enc_mask=enc_mask, num_tokens_to_generate=10
        )

        return {'loss': loss, 'predicted_token_ids': predicted_token_ids, 'labels': labels}

    def inference_epoch_end(self, outputs):
        losses = [x['loss'] for x in outputs]
        averaged_loss = average_losses_across_data_parallel_group(losses)
        all_preds = []
        all_labels = []
        for item in outputs:
            preds = item['predicted_token_ids'].cpu().numpy().tolist()
            labels = item['labels'].cpu().numpy().tolist()
            for i, (pred, label) in enumerate(zip(preds, labels)):
                if self.model.tokenizer.eos_id in pred:
                    idx = pred.index(self.model.tokenizer.eos_id)
                    pred = pred[:idx]
                pred = self.model.tokenizer.ids_to_text(pred)
                label = self.model.tokenizer.ids_to_text(label)
                all_preds.append(pred)
                all_labels.append(label)

        correct = 0
        for pred, label in zip(all_preds, all_labels):
            if pred == label:
                correct += 1
        acc = correct / len(all_preds)
        return averaged_loss[0], acc

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        val_loss, val_acc = self.inference_epoch_end(outputs)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)
        logging.info(f'Validation loss: {val_loss}')
        logging.info(f'Validation accuracy: {val_acc}')

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        test_loss, test_acc = self.inference_epoch_end(outputs)
        self.log('test_loss',test_loss, prog_bar=True)
        self.log('test_acc', test_acc, prog_bar=True)
        logging.info(f'Test loss: {test_loss}')
        logging.info(f'Test accuracy: {test_acc}')

    def build_train_valid_test_datasets(self, test_only=False):
        logging.info('Building GLUE datasets.')
        self._test_ds = TextToTextGLUEDataset(
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
        self._validation_ds = TextToTextGLUEDataset(
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

    def build_pretraining_data_loader(self, dataset, batch_size, shuffle, num_workers, pin_memory):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    def setup(self, stage=None):
        if stage == 'predict':
            return
        self.build_train_valid_test_datasets(test_only=stage=='test')
        self.setup_test_data()
        if stage == 'test':
            return
        self.setup_training_data()
        self.setup_validation_data()

    def setup_training_data(self):
        self._train_dl = self.build_pretraining_data_loader(
            self._train_ds,
            self.cfg.data.train_ds.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.train_ds.num_workers,
            pin_memory=True,
        )

    def setup_validation_data(self):
        self._validation_dl = self.build_pretraining_data_loader(
            self._validation_ds,
            self.cfg.data.validation_ds.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.validation_ds.num_workers,
            pin_memory=True,
        )

    def setup_test_data(self):
        self._test_dl = self.build_pretraining_data_loader(
            self._test_ds,
            self.cfg.data.test_ds.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.test_ds.num_workers,
            pin_memory=True,
        )

    def list_available_models():
        pass
