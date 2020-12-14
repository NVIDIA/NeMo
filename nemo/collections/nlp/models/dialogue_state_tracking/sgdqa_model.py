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

from nemo.collections.nlp.data import SGDDataset
from nemo.collections.nlp.data.dialogue_state_tracking_sgd import Schema, SGDDataProcessor
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
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
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
        
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
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

        avg_loss = torch.stack([x[f'{prefix}_loss'] for x in outputs]).mean()

        example_id_num = torch.cat([x[f'{prefix}_tensors']['example_id_num'] for x in outputs])
        service_id = torch.cat([x[f'{prefix}_tensors']['service_id'] for x in outputs])
        is_real_example = torch.cat([x[f'{prefix}_tensors']['is_real_example'] for x in outputs])
        logit_intent_status = torch.cat([x[f'{prefix}_tensors']['logit_intent_status'] for x in outputs])
        logit_req_slot_status = torch.cat([x[f'{prefix}_tensors']['logit_req_slot_status'] for x in outputs])
        logit_cat_slot_status = torch.cat([x[f'{prefix}_tensors']['logit_cat_slot_status'] for x in outputs])
        logit_cat_slot_value_status = torch.cat([x[f'{prefix}_tensors']['logit_cat_slot_value_status'] for x in outputs])
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

        import ipdb; ipdb.set_trace()
        exact_match, f1, all_predictions, all_nbest = -1, -1, [], []
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            unique_ids = []
            start_logits = []
            end_logits = []
            for u in all_unique_ids:
                unique_ids.extend(tensor2list(u))
            for u in all_start_logits:
                start_logits.extend(tensor2list(u))
            for u in all_end_logits:
                end_logits.extend(tensor2list(u))

            eval_dataset = self._test_dl.dataset if self.testing else self._validation_dl.dataset
            exact_match, f1, all_predictions, all_nbest = eval_dataset.evaluate(
                unique_ids=unique_ids,
                start_logits=start_logits,
                end_logits=end_logits,
                n_best_size=self._cfg.dataset.n_best_size,
                max_answer_length=self._cfg.dataset.max_answer_length,
                version_2_with_negative=self._cfg.dataset.version_2_with_negative,
                null_score_diff_threshold=self._cfg.dataset.null_score_diff_threshold,
                do_lower_case=self._cfg.dataset.do_lower_case,
            )

        logging.info(f"{prefix} exact match {exact_match}")
        logging.info(f"{prefix} f1 {f1}")

        self.log(f'{prefix}_loss', avg_loss)
        self.log(f'{prefix}_exact_match', exact_match)
        self.log(f'{prefix}_f1', f1)

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
        """
        Setup dataloader from config
        Args:
            cfg: config for the dataloader
        Return:
            Pytorch Dataloader
        """
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
