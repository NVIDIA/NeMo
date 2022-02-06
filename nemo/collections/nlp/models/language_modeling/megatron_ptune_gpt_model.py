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

import os
from typing import Any, Dict, Optional, Union

import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.glue_benchmark.gpt_ptune_dataset import GPTPTuneDataset
from nemo.collections.nlp.models.language_modeling.megatron.t5_model import t5_position_ids
from nemo.collections.nlp.data.language_modeling.megatron.t5_dataset import (
    make_attention_mask_3d,
    make_history_mask_3d,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.utils import logging
from nemo.collections.nlp.modules.common.prompt_encoder import PromptEncoder
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
import torch.nn as nn
try:
    from apex.transformer import tensor_parallel
    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False


__all__ = ['MegatronGPTPTuneModel']

NUM_TOKEN_TO_GEN = 10

class MegatronGPTPTuneModel(NLPModel):
    """
    Megatron GPT P-Tune
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        initialize_model_parallel_for_nemo(
            world_size=trainer.world_size,
            global_rank=trainer.global_rank,
            local_rank=trainer.local_rank,
            tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
            seed=cfg.get('seed', 1234),
        )

        # shared params for dataset and data loaders
        # tokenizer needs to get initialized before the super.__init__()
        # as dataloaders and datasets need it to process the data

        self.model = MegatronGPTModel.restore_from(
            self.register_artifact('language_model.nemo_file', cfg.language_model.get('nemo_file', None)),
            trainer=trainer)

        self.tokenizer = self.model.tokenizer

        if not cfg.use_lm_finetune:
            self.model.freeze()

        hidden_size = self.model.cfg.hidden_size

        # register the file containing the labels into the artifacts to get stored in the '.nemo' file later
        self.embeddings = self.model.model.language_model.embedding.word_embeddings

        # self.vocab = self.tokenizer.tokenizer.get_vocab()

        self.template = cfg.prompt_encoder.template

        self.prompt_encoder = PromptEncoder(
            template=cfg.prompt_encoder.template,
            hidden_size=hidden_size,
            lstm_dropout=cfg.prompt_encoder.dropout,
            num_layers=cfg.prompt_encoder.num_layers,
        )

        # load prompt encoder
        self.hidden_size = hidden_size
        self.tokenizer.add_special_tokens({'additional_special_tokens': [cfg.pseudo_token]})

        self.pseudo_token_id = self.tokenizer.token_to_id(cfg.pseudo_token)
        self.pad_token_id = (
            self.tokenizer.pad_id
            if self.tokenizer.pad_id is not None
            else self.tokenizer.unk_id
        )
        self.spell_length = sum(self.template)

    def embed_input(self, queries):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()

        queries_for_embedding[(queries == self.pseudo_token_id)] = self.pad_token_id
        raw_embeds = self.embeddings(queries_for_embedding).clone()

        blocked_indices = (
            (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]
        )  # bz
        replace_embeds = self.prompt_encoder()
        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds

    def get_loss(self, batch):
        enc_input = batch['enc_input']
        labels = batch['labels']
        loss_mask = batch['loss_mask']
        enc_query = batch['enc_query']
        input_attn_mask = batch['input_attn_mask']

        input_attn_mask = input_attn_mask.unsqueeze(1) < 0.5

        input_embeds = self.embed_input(enc_input)

        encoder_position_ids = t5_position_ids(enc_input)

        position_embeddings = self.model.model.language_model.embedding.position_embeddings(encoder_position_ids)

        encoder_input = input_embeds + position_embeddings

        dtype = self.model.model.language_model.encoder.layers[0].dtype

        if dtype == torch.float32:
            output = self.model.model(
                None,
                None,
                encoder_input=encoder_input,
                attention_mask=input_attn_mask,
                labels=labels,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=dtype):
                output = self.model.model(
                    None,
                    None,
                    encoder_input=encoder_input,
                    attention_mask=input_attn_mask,
                    labels=labels,
                )
        output_tensor, encoder_hidden_states = output

        # _, returned_pred = self.get_prediction(batch_size, label_position, logits)
        # returned_label = self.get_ground_truth_labels(batch_size, label_ids)
        # return floss, returned_pred, returned_label

        loss = self.loss_func(loss_mask, output_tensor)
        return loss

    def loss_func(self, loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        # TODO: add nemo version here
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()  # sequence level nll
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
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

    def inference_step(self, batch, batch_ix):
        loss = self.get_loss(batch)
        enc_query = batch['enc_query']
        labels = batch['labels']
        label_position = batch['label_position']
        # loss, tokens_enc, labels, enc_mask, encoder_input = self.get_loss(batch)
        predicted_token_ids, log_probs = self.decode(
            enc_query=enc_query, label_position=label_position, num_tokens_to_generate=NUM_TOKEN_TO_GEN
        )

        return {'loss': loss, 'predicted_token_ids': predicted_token_ids, 'labels': labels, 'label_position': label_position}

    def decode(self, enc_query, label_position, num_tokens_to_generate):
        predicted_tokens_dec = enc_query

        label_start = label_position[:, 0].clone()

        for _ in range(num_tokens_to_generate):
            attn_mask = make_attention_mask_3d(
                predicted_tokens_dec, predicted_tokens_dec, self.pad_token_id
            )
            attn_mask = attn_mask * make_history_mask_3d(predicted_tokens_dec)

            attn_mask = attn_mask < 0.5

            attn_mask = attn_mask.unsqueeze(1)

            input_embeds = self.embed_input(predicted_tokens_dec)

            encoder_position_ids = t5_position_ids(predicted_tokens_dec)
            position_embeddings = self.model.model.language_model.embedding.position_embeddings(encoder_position_ids)

            encoder_input = input_embeds + position_embeddings

            dtype = self.model.model.language_model.encoder.layers[0].dtype

            if dtype == torch.float32:
                output = self.model.model(
                    None,
                    None,
                    encoder_input=encoder_input,
                    attention_mask=attn_mask,
                )
            else:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    output = self.model.model(
                        None,
                        None,
                        encoder_input=encoder_input,
                        attention_mask=attn_mask,
                    )
            output_tensor = output

            output_tensor = tensor_parallel.gather_from_tensor_model_parallel_region(output_tensor)


            # only use the allowed vocab if it is defined 
            log_probs, token_ids = torch.max(nn.functional.log_softmax(output_tensor, dim=-1), dim=-1)

            # append empty array in the end
            # predicted_tokens_dec = torch.cat([predicted_tokens_dec, token_ids[:, -1].unsqueeze(1)], 1)
            # new_pred = torch.zeros_like(token_ids[:, 0:1])
            new_pred = torch.full_like(token_ids[:, 0:1], self.pad_token_id)
            predicted_tokens_dec = torch.cat([predicted_tokens_dec, new_pred], 1)

            predicted = torch.gather(token_ids, 1, label_start.view(-1, 1))

            # need to scatter the token id at the right position
            label_start += 1
            predicted_tokens_dec.scatter_(1, label_start.view(-1, 1), predicted)

        return predicted_tokens_dec, log_probs

    def inference_epoch_end(self, outputs):
        losses = [x['loss'] for x in outputs]
        averaged_loss = average_losses_across_data_parallel_group(losses)
        all_preds = []
        all_labels = []
        special_tokens = set([self.tokenizer.eos_id, 
                              self.tokenizer.pad_id, 
                              self.tokenizer.sep_id, 
                              self.tokenizer.unk_id,
                              self.tokenizer.bos_id,
                              self.tokenizer.cls_id])
        for item in outputs:
            preds = item['predicted_token_ids'].cpu().numpy().tolist()
            labels = item['labels'].cpu().numpy().tolist()
            label_positions = item['label_position'].cpu().numpy().tolist()
            for i, (pred, label, label_position) in enumerate(zip(preds, labels, label_positions)):
                start_position = label_position[0]+1
                pred = pred[start_position:]
                if self.tokenizer.eos_id in pred:
                    idx = pred.index(self.tokenizer.eos_id)
                    pred = pred[:idx]
                pred = [id for id in pred if id not in special_tokens]
                label = [id for id in label[label_position[0]:label_position[1]] if id not in special_tokens]
                pred = self.tokenizer.ids_to_text(pred)
                label = self.tokenizer.ids_to_text(label)                
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
        self._test_ds = GPTPTuneDataset(
            self.cfg.data.test_ds.file_path,
            task_name=self.cfg.data.test_ds.task_name,
            data_type="test",
            tokenizer=self.tokenizer,
            templates=self.template,
            pseudo_token_id=self.pseudo_token_id,
            pad_id=self.pad_token_id,
            max_seq_length=self.model.cfg.encoder_seq_length,
        )
        if test_only:
            return None, None, self._test_ds
        self._train_ds = GPTPTuneDataset(
            self.cfg.data.train_ds.file_path,
            task_name=self.cfg.data.train_ds.task_name,
            data_type="train",
            tokenizer=self.tokenizer,
            templates=self.template,
            pseudo_token_id=self.pseudo_token_id,
            pad_id=self.pad_token_id,
            max_seq_length=self.model.cfg.encoder_seq_length,
        )
        self._validation_ds = GPTPTuneDataset(
            self.cfg.data.validation_ds.file_path,
            task_name=self.cfg.data.validation_ds.task_name,
            data_type="validation",
            tokenizer=self.tokenizer,
            templates=self.template,
            pseudo_token_id=self.pseudo_token_id,
            pad_id=self.pad_token_id,
            max_seq_length=self.model.cfg.encoder_seq_length,
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