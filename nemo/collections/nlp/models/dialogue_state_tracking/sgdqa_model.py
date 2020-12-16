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

import os
from typing import Dict, List, Optional, Union

import onnx
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

import nemo.collections.nlp.data.dialogue_state_tracking_sgd.prediction_utils as pred_utils
from nemo.collections.nlp.data import Schema, SGDDataProcessor, SGDDataset
from nemo.collections.nlp.data.dialogue_state_tracking_sgd.evaluate import evaluate, get_in_domain_services
from nemo.collections.nlp.losses import SGDDialogueStateLoss
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules import SGDDecoder, SGDEncoder
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import NeuralType
from nemo.utils import logging

__all__ = ['SGDQAModel']


def tensor2list(tensor):
    return tensor.detach().cpu().tolist()


def get_str_example_id(eval_dataset, ids_to_service_names_dict, example_id_num):
    def format_turn_id(ex_id_num):
        dialog_id_1, dialog_id_2, turn_id, service_id, model_task_id, slot_intent_id, value_id = ex_id_num
        return "{}-{}_{:05d}-{:02d}-{}-{}-{}-{}".format(
            eval_dataset,
            dialog_id_1,
            dialog_id_2,
            turn_id,
            ids_to_service_names_dict[service_id],
            model_task_id,
            slot_intent_id,
            value_id,
        )

    return list(map(format_turn_id, tensor2list(example_id_num)))


def combine_predictions_in_example(predictions, batch_size):
    '''
    Combines predicted values to a single example. 
    Dict: sample idx-> keys-> values
    '''
    examples_preds = [{} for _ in range(batch_size)]
    for k, v in predictions.items():
        if k != 'example_id':
            v = torch.chunk(v, batch_size)

        for i in range(batch_size):
            if k == 'example_id':
                examples_preds[i][k] = v[i]
            else:
                examples_preds[i][k] = v[i].view(-1)
    return examples_preds


class SGDQAModel(NLPModel):
    """Dialogue State Tracking Model SGD-QA"""

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.decoder.output_types

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        self.setup_tokenizer(cfg.tokenizer)
        super().__init__(cfg=cfg, trainer=trainer)
        self.bert_model = get_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            config_file=cfg.language_model.config_file,
            config_dict=OmegaConf.to_container(cfg.language_model.config) if cfg.language_model.config else None,
            checkpoint_file=cfg.language_model.lm_checkpoint,
        )

        self.encoder = SGDEncoder(hidden_size=self.bert_model.config.hidden_size, dropout=self._cfg.encoder.dropout)
        self.decoder = SGDDecoder(embedding_dim=self.bert_model.config.hidden_size)
        self.loss = SGDDialogueStateLoss(reduction="mean")

    @typecheck()
    def forward(self, input_ids, token_type_ids, attention_mask):
        token_embeddings = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        encoded_utterance, token_embeddings = self.encoder(hidden_states=token_embeddings)
        (
            logit_intent_status,
            logit_req_slot_status,
            logit_cat_slot_status,
            logit_cat_slot_value_status,
            logit_noncat_slot_status,
            logit_noncat_slot_start,
            logit_noncat_slot_end,
        ) = self.decoder(
            encoded_utterance=encoded_utterance, token_embeddings=token_embeddings, utterance_mask=attention_mask
        )
        return (
            logit_intent_status,
            logit_req_slot_status,
            logit_cat_slot_status,
            logit_cat_slot_value_status,
            logit_noncat_slot_status,
            logit_noncat_slot_start,
            logit_noncat_slot_end,
        )

    def training_step(self, batch, batch_idx):
        (
            example_id_num,
            service_id,
            is_real_example,
            utterance_ids,
            token_type_ids,
            attention_mask,
            intent_status,
            requested_slot_status,
            categorical_slot_status,
            categorical_slot_value_status,
            noncategorical_slot_status,
            noncategorical_slot_value_start,
            noncategorical_slot_value_end,
            start_char_idx,
            end_char_idx,
            task_mask,
        ) = batch
        (
            logit_intent_status,
            logit_req_slot_status,
            logit_cat_slot_status,
            logit_cat_slot_value_status,
            logit_noncat_slot_status,
            logit_noncat_slot_start,
            logit_noncat_slot_end,
        ) = self(input_ids=utterance_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        loss = self.loss(
            logit_intent_status=logit_intent_status,
            intent_status=intent_status,
            logit_req_slot_status=logit_req_slot_status,
            requested_slot_status=requested_slot_status,
            logit_cat_slot_status=logit_cat_slot_status,
            categorical_slot_status=categorical_slot_status,
            logit_cat_slot_value_status=logit_cat_slot_value_status,
            categorical_slot_value_status=categorical_slot_value_status,
            logit_noncat_slot_status=logit_noncat_slot_status,
            noncategorical_slot_status=noncategorical_slot_status,
            logit_noncat_slot_start=logit_noncat_slot_start,
            logit_noncat_slot_end=logit_noncat_slot_end,
            noncategorical_slot_value_start=noncategorical_slot_value_start,
            noncategorical_slot_value_end=noncategorical_slot_value_end,
            task_mask=task_mask,
        )
        lr = self._optimizer.param_groups[0]['lr']

        self.log('train_loss', loss)
        self.log('lr', lr, prog_bar=True)

        return {
            'loss': loss,
            'lr': lr,
        }

    def validation_step(self, batch, batch_idx):
        prefix = 'val'
        (
            example_id_num,
            service_id,
            is_real_example,
            utterance_ids,
            token_type_ids,
            attention_mask,
            intent_status,
            requested_slot_status,
            categorical_slot_status,
            categorical_slot_value_status,
            noncategorical_slot_status,
            noncategorical_slot_value_start,
            noncategorical_slot_value_end,
            start_char_idx,
            end_char_idx,
            task_mask,
        ) = batch
        (
            logit_intent_status,
            logit_req_slot_status,
            logit_cat_slot_status,
            logit_cat_slot_value_status,
            logit_noncat_slot_status,
            logit_noncat_slot_start,
            logit_noncat_slot_end,
        ) = self(input_ids=utterance_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        loss = self.loss(
            logit_intent_status=logit_intent_status,
            intent_status=intent_status,
            logit_req_slot_status=logit_req_slot_status,
            requested_slot_status=requested_slot_status,
            logit_cat_slot_status=logit_cat_slot_status,
            categorical_slot_status=categorical_slot_status,
            logit_cat_slot_value_status=logit_cat_slot_value_status,
            categorical_slot_value_status=categorical_slot_value_status,
            logit_noncat_slot_status=logit_noncat_slot_status,
            noncategorical_slot_status=noncategorical_slot_status,
            logit_noncat_slot_start=logit_noncat_slot_start,
            logit_noncat_slot_end=logit_noncat_slot_end,
            noncategorical_slot_value_start=noncategorical_slot_value_start,
            noncategorical_slot_value_end=noncategorical_slot_value_end,
            task_mask=task_mask,
        )

        tensors = {
            'example_id_num': example_id_num,
            'service_id': service_id,
            'is_real_example': is_real_example,
            'logit_intent_status': logit_intent_status,
            'logit_req_slot_status': logit_req_slot_status,
            'logit_cat_slot_status': logit_cat_slot_status,
            'logit_cat_slot_value_status': logit_cat_slot_value_status,
            'logit_noncat_slot_status': logit_noncat_slot_status,
            'logit_noncat_slot_start': logit_noncat_slot_start,
            'logit_noncat_slot_end': logit_noncat_slot_end,
            'start_char_idx': start_char_idx,
            'end_char_idx': end_char_idx,
        }
        self.log(f'{prefix}_loss', loss)
        return {f'{prefix}_loss': loss, f'{prefix}_tensors': tensors}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        outputs: list of individual outputs of each validation step.
        """

        prefix = 'val'
        eval_dataset = 'dev'

        avg_loss = torch.stack([x[f'{prefix}_loss'] for x in outputs]).mean()

        example_id_num = torch.cat([x[f'{prefix}_tensors']['example_id_num'] for x in outputs])
        service_id = torch.cat([x[f'{prefix}_tensors']['service_id'] for x in outputs])
        is_real_example = torch.cat([x[f'{prefix}_tensors']['is_real_example'] for x in outputs])
        logit_intent_status = torch.cat([x[f'{prefix}_tensors']['logit_intent_status'] for x in outputs])
        logit_req_slot_status = torch.cat([x[f'{prefix}_tensors']['logit_req_slot_status'] for x in outputs])
        logit_cat_slot_status = torch.cat([x[f'{prefix}_tensors']['logit_cat_slot_status'] for x in outputs])
        logit_cat_slot_value_status = torch.cat(
            [x[f'{prefix}_tensors']['logit_cat_slot_value_status'] for x in outputs]
        )
        logit_noncat_slot_status = torch.cat([x[f'{prefix}_tensors']['logit_noncat_slot_status'] for x in outputs])
        logit_noncat_slot_start = torch.cat([x[f'{prefix}_tensors']['logit_noncat_slot_start'] for x in outputs])
        logit_noncat_slot_end = torch.cat([x[f'{prefix}_tensors']['logit_noncat_slot_end'] for x in outputs])
        start_char_idx = torch.cat([x[f'{prefix}_tensors']['start_char_idx'] for x in outputs])
        end_char_idx = torch.cat([x[f'{prefix}_tensors']['end_char_idx'] for x in outputs])

        all_example_id_num = []
        all_service_id = []
        all_is_real_example = []
        all_logit_intent_status = []
        all_logit_req_slot_status = []
        all_logit_cat_slot_status = []
        all_logit_cat_slot_value_status = []
        all_logit_noncat_slot_status = []
        all_logit_noncat_slot_start = []
        all_logit_noncat_slot_end = []
        all_start_char_idx = []
        all_end_char_idx = []

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            for ind in range(world_size):
                all_example_id_num.append(torch.empty_like(example_id_num))
                all_service_id.append(torch.empty_like(service_id))
                all_is_real_example.append(torch.empty_like(is_real_example))
                all_logit_intent_status.append(torch.empty_like(logit_intent_status))
                all_logit_req_slot_status.append(torch.empty_like(logit_req_slot_status))
                all_logit_cat_slot_status.append(torch.empty_like(logit_cat_slot_status))
                all_logit_cat_slot_value_status.append(torch.empty_like(logit_cat_slot_value_status))
                all_logit_noncat_slot_status.append(torch.empty_like(logit_noncat_slot_status))
                all_logit_noncat_slot_start.append(torch.empty_like(logit_noncat_slot_start))
                all_logit_noncat_slot_end.append(torch.empty_like(logit_noncat_slot_end))
                all_start_char_idx.append(torch.empty_like(start_char_idx))
                all_end_char_idx.append(torch.empty_like(end_char_idx))

            torch.distributed.all_gather(all_example_id_num, example_id_num)
            torch.distributed.all_gather(all_service_id, service_id)
            torch.distributed.all_gather(all_is_real_example, is_real_example)
            torch.distributed.all_gather(all_logit_intent_status, logit_intent_status)
            torch.distributed.all_gather(all_logit_req_slot_status, logit_req_slot_status)
            torch.distributed.all_gather(all_logit_cat_slot_status, logit_cat_slot_status)
            torch.distributed.all_gather(all_logit_cat_slot_value_status, logit_cat_slot_value_status)
            torch.distributed.all_gather(all_logit_noncat_slot_status, logit_noncat_slot_status)
            torch.distributed.all_gather(all_logit_noncat_slot_start, logit_noncat_slot_start)
            torch.distributed.all_gather(all_logit_noncat_slot_end, logit_noncat_slot_end)
            torch.distributed.all_gather(all_start_char_idx, start_char_idx)
            torch.distributed.all_gather(all_end_char_idx, end_char_idx)
        else:
            all_example_id_num.append(example_id_num)
            all_service_id.append(service_id)
            all_is_real_example.append(is_real_example)
            all_logit_intent_status.append(logit_intent_status)
            all_logit_req_slot_status.append(logit_req_slot_status)
            all_logit_cat_slot_status.append(logit_cat_slot_status)
            all_logit_cat_slot_value_status.append(logit_cat_slot_value_status)
            all_logit_noncat_slot_status.append(logit_noncat_slot_status)
            all_logit_noncat_slot_start.append(logit_noncat_slot_start)
            all_logit_noncat_slot_end.append(logit_noncat_slot_end)
            all_start_char_idx.append(start_char_idx)
            all_end_char_idx.append(end_char_idx)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            ids_to_service_names_dict = self.dialogues_processor.schemas._services_id_to_vocab
            example_id = get_str_example_id(self._validation_dl.dataset, ids_to_service_names_dict, example_id_num)
            intent_status = torch.nn.Sigmoid()(logit_intent_status)

            # Scores are output for each requested slot.
            req_slot_status = torch.nn.Sigmoid()(logit_req_slot_status)

            # For categorical slots, the status of each slot and the predicted value are output.
            cat_slot_status_dist = torch.nn.Softmax(dim=-1)(logit_cat_slot_status)

            cat_slot_status = torch.argmax(logit_cat_slot_status, axis=-1)
            cat_slot_status_p = cat_slot_status_dist
            cat_slot_value_status = torch.nn.Sigmoid()(logit_cat_slot_value_status)

            # For non-categorical slots, the status of each slot and the indices for spans are output.
            noncat_slot_status_dist = torch.nn.Softmax(dim=-1)(logit_noncat_slot_status)

            noncat_slot_status = torch.argmax(logit_noncat_slot_status, axis=-1)
            noncat_slot_status_p = noncat_slot_status_dist

            softmax = torch.nn.Softmax(dim=-1)
            start_scores = softmax(logit_noncat_slot_start)
            end_scores = softmax(logit_noncat_slot_end)

            batch_size, max_num_tokens = end_scores.size()
            # Find the span with the maximum sum of scores for start and end indices.
            total_scores = torch.unsqueeze(start_scores, axis=2) + torch.unsqueeze(end_scores, axis=1)
            # Mask out scores where start_index > end_index.
            # device = total_scores.get_device()
            start_idx = torch.arange(max_num_tokens, device=total_scores.get_device()).view(1, -1, 1)
            end_idx = torch.arange(max_num_tokens, device=total_scores.get_device()).view(1, 1, -1)
            invalid_index_mask = (start_idx > end_idx).repeat(batch_size, 1, 1)
            total_scores = torch.where(
                invalid_index_mask,
                torch.zeros(total_scores.size(), device=total_scores.get_device(), dtype=total_scores.dtype),
                total_scores,
            )
            max_span_index = torch.argmax(total_scores.view(-1, max_num_tokens ** 2), axis=-1)
            max_span_p = torch.max(total_scores.view(-1, max_num_tokens ** 2), axis=-1)[0]
            noncat_slot_p = max_span_p

            span_start_index = torch.div(max_span_index, max_num_tokens)
            span_end_index = torch.fmod(max_span_index, max_num_tokens)

            noncat_slot_start = span_start_index
            noncat_slot_end = span_end_index

            # Add inverse alignments.
            noncat_alignment_start = start_char_idx
            noncat_alignment_end = end_char_idx

            in_domain_services = get_in_domain_services(
                os.path.join(self._cfg.dataset.data_dir, eval_dataset, "schema.json"),
                self.dialogues_processor.get_seen_services("train"),
            )
            ##############
            # we'll write predictions to file in Dstc8/SGD format during evaluation callback
            prediction_dir = self.self.trainer.log_dir
            prediction_dir = os.path.join(
                prediction_dir, 'predictions', 'pred_res_{}_{}'.format(eval_dataset, self._cfg.dataset.task_name)
            )
            os.makedirs(prediction_dir, exist_ok=True)

            input_json_files = SGDDataProcessor.get_dialogue_files(
                self._cfg.dataset.data_dir, eval_dataset, self._cfg.dataset.task_name
            )

            predictions = {}
            predictions['example_id'] = example_id
            predictions['service_id'] = service_id
            predictions['is_real_example'] = is_real_example
            predictions['intent_status'] = intent_status
            predictions['req_slot_status'] = req_slot_status
            predictions['cat_slot_status'] = cat_slot_status
            predictions['cat_slot_status_p'] = cat_slot_status_p
            predictions['cat_slot_value_status'] = cat_slot_value_status
            predictions['noncat_slot_status'] = noncat_slot_status
            predictions['noncat_slot_status_p'] = noncat_slot_status_p
            predictions['noncat_slot_p'] = noncat_slot_p
            predictions['noncat_slot_start'] = noncat_slot_start
            predictions['noncat_slot_end'] = noncat_slot_end
            predictions['noncat_alignment_start'] = noncat_alignment_start
            predictions['noncat_alignment_end'] = noncat_alignment_end

            predictions = combine_predictions_in_example(predictions, service_id.shape[0])
            pred_utils.write_predictions_to_file(
                predictions,
                input_json_files,
                output_dir=prediction_dir,
                schemas=self.dialogues_processor.schemas,
                state_tracker=self._cfg.dataset.state_tracker,
                eval_debug=False,
                in_domain_services=in_domain_services,
                cat_value_thresh=0.0,
                non_cat_value_thresh=0.0,
                probavg=False,
            )
            metrics = evaluate(
                prediction_dir,
                self._cfg.dataset.data_dir,
                eval_dataset,
                in_domain_services,
                joint_acc_across_turn=False,
                no_fuzzy_match=False,
            )

        self.log(f'{prefix}_loss', avg_loss, prog_bar=True)

    def prepare_data(self):
        schema_config = {
            "MAX_NUM_CAT_SLOT": 6,
            "MAX_NUM_NONCAT_SLOT": 12,
            "MAX_NUM_VALUE_PER_CAT_SLOT": 12,
            "MAX_NUM_INTENT": 4,
            "NUM_TASKS": 6,
            "MAX_SEQ_LENGTH": self._cfg.dataset.max_seq_length,
        }
        all_schema_json_paths = []
        for dataset_split in ['train', 'test', 'dev']:
            all_schema_json_paths.append(os.path.join(self._cfg.dataset.data_dir, dataset_split, "schema.json"))
        schemas = Schema(all_schema_json_paths)
        self.dialogues_processor = SGDDataProcessor(
            task_name=self._cfg.dataset.task_name,
            data_dir=self._cfg.dataset.data_dir,
            dialogues_example_dir=self._cfg.dataset.dialogues_example_dir,
            tokenizer=self.tokenizer,
            schemas=schemas,
            schema_config=schema_config,
            subsample=self._cfg.dataset.subsample,
            overwrite_dial_files=not self._cfg.dataset.use_cache,
        )

    def setup_training_data(self, train_data_config: Optional[DictConfig] = None):
        self.prepare_data()
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, split='train')

    def setup_validation_data(self, val_data_config: Optional[DictConfig] = None):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, split='dev')

    def _setup_dataloader_from_config(self, cfg: DictConfig, split: str) -> DataLoader:
        dataset_cfg = self._cfg.dataset
        data_dir = dataset_cfg.data_dir

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory is not found at: {data_dir}.")

        dataset = SGDDataset(dataset_split=split, dialogues_processor=self.dialogues_processor)

        dl = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            collate_fn=dataset.collate_fn,
            drop_last=cfg.drop_last,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
        return dl

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        pass
