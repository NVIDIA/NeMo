# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.metrics.classification_accuracy import ExactStringPerCategoryMatchMetric
from nemo.collections.nlp.data.glue_benchmark.glue_benchmark_dataset import (
    TextToTextGLUEDataset,
    TextToTextXNlIDataset,
)
from nemo.collections.nlp.models.language_modeling.megatron_glue_model import MegatronT5GLUEModel
from nemo.utils import logging

__all__ = ['MegatronT5XNLIModel']


class MegatronT5XNLIModel(MegatronT5GLUEModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg=cfg, trainer=trainer)
        self.cfg = cfg
        self.acc_metrics = ExactStringPerCategoryMatchMetric(self.cfg.eval_languages)

    def process_micro_batch(self, batch):
        """Process a microbatch."""
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask = super().process_micro_batch(batch)
        if 'lang' in batch:
            lang = batch['lang']
            return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, lang
        else:
            return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask

    def process_global_batch(self, global_batch):
        """Process a list of microbatches into a global batch."""
        # If there is no language information in the global batch (ex: English MNLI), we can use the parent global batch processor as is.
        if len(global_batch[0]) == 6:
            return super().process_global_batch(global_batch)

        # For validation data (XNLI), we need to process the global batch and and then deal with language info separately.
        else:
            assert len(global_batch[0]) == 7
            langs_list = []
            (
                tokens_enc_tensor,
                tokens_dec_tensor,
                loss_mask_tensor,
                labels_tensor,
                enc_mask_tensor,
                dec_mask_tensor,
            ) = super().process_global_batch(
                [{k: v for k, v in micro_batch.items() if k != 'lang'} for micro_batch in global_batch]
            )
            for micro_batch in global_batch:
                langs_list.extend(micro_batch['lang'])
            return (
                tokens_enc_tensor,
                tokens_dec_tensor,
                loss_mask_tensor,
                labels_tensor,
                enc_mask_tensor,
                dec_mask_tensor,
                langs_list,
            )

    def inference_step(self, batch, batch_idx):
        # Remove languages from the global batch before computing the loss. Langs is a list and not a tensor will mess with apex fwd/bwd calls.
        processed_batch = []
        for micro_batch in batch:
            micro_batch = {k: v for k, v in micro_batch.items() if k != 'lang'}
            processed_batch.append(micro_batch)

        # Call the parent's parent because the parent inference step will compute accuracy metrics.
        loss = super(MegatronT5GLUEModel, self).validation_step(
            processed_batch, batch_idx, reconfigure_microbatch_size=True
        )

        tokens_enc, _, _, labels, enc_mask, _, langs = self.process_global_batch(batch)

        predicted_token_ids, _ = self.decode(tokens_enc=tokens_enc, enc_mask=enc_mask, num_tokens_to_generate=10)

        preds_text, labels_text = self.preds_and_labels_to_text(predicted_token_ids, labels)
        for _, (pred, label, lang) in enumerate(zip(preds_text, labels_text, langs)):
            _ = self.acc_metrics(pred, label, lang)

        return loss

    def inference_epoch_end(self, outputs, mode):
        averaged_loss, _ = super().inference_epoch_end(outputs, mode)
        acc_result = self.acc_metrics.compute()
        self.log(f'{mode}_loss', averaged_loss)
        self.log(f'{mode}_acc', acc_result['acc'])
        for lang in self.cfg.eval_languages:
            self.log(f'{lang}_acc', acc_result[lang])
            logging.info(f"{mode} {lang} accuracy: {acc_result[lang]} total: {acc_result[lang+'_total']}")
        logging.info(f"{mode} accuracy: {acc_result['acc']}")
        self.acc_metrics.reset()
        return averaged_loss, acc_result

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)

    def setup(self, stage=None):
        # This is just to keep the parent class happy since we override its setup() method.
        self.init_consumed_samples = 0

        if stage == 'predict':
            return

        # NOTE: PTL uses the same stage string "test" for both testing and validation.
        self.build_train_valid_test_datasets(test_only=stage == 'test')
        self.setup_test_data()
        if stage == 'test':
            return
        self.setup_validation_data()
        self.setup_training_data()

    def setup_test_data(self):
        self._test_dl = self.build_data_loader(
            self._test_ds,
            micro_batch_size=self.cfg.data.test_ds.micro_batch_size,
            global_batch_size=self.cfg.data.test_ds.global_batch_size,
            shuffle=self.cfg.data.test_ds.shuffle,
            num_workers=self.cfg.data.test_ds.num_workers,
            pin_memory=self.cfg.data.test_ds.pin_memory,
            drop_last=self.cfg.data.test_ds.drop_last,
            check_validation_interval=False,
        )

    def validation_epoch_end(self, outputs):
        val_loss, val_acc = self.inference_epoch_end(outputs, 'validation')

    def test_epoch_end(self, outputs):
        test_loss, test_acc = self.inference_epoch_end(outputs, 'test')

    def build_train_valid_test_datasets(self, test_only=False):
        logging.info('Building XNLI datasets.')
        self._test_ds = TextToTextXNlIDataset(
            self.cfg.data.test_ds.file_path,
            task_name=self.cfg.data.test_ds.task_name,
            tokenizer=self.tokenizer,
            max_seq_length=self.cfg.data.test_ds.max_seq_length,
            lang_list=self.cfg.eval_languages,
        )
        if test_only:
            return None, None, self._test_ds
        self._train_ds = TextToTextGLUEDataset(
            self.cfg.data.train_ds.file_path,
            task_name=self.cfg.data.train_ds.task_name,
            tokenizer=self.tokenizer,
            max_seq_length=self.cfg.data.train_ds.max_seq_length,
        )
        self._validation_ds = TextToTextXNlIDataset(
            self.cfg.data.validation_ds.file_path,
            task_name=self.cfg.data.validation_ds.task_name,
            tokenizer=self.tokenizer,
            max_seq_length=self.cfg.data.validation_ds.max_seq_length,
            lang_list=self.cfg.eval_languages,
        )
        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building XNLI datasets.')
        return self._train_ds, self._validation_ds, self._test_ds
