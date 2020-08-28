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

import json
from typing import Dict, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from nemo.collections.common.losses import SpanningLoss
from nemo.collections.common.tokenizers.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.data import SquadDataset
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.common_utils import get_pretrained_lm_model
from nemo.core.classes import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import NeuralType
from nemo.utils import logging

__all__ = ['QAModel']


class QAModel(ModelPT):
    """
    BERT encoder with QA head training.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.classifier.output_types

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.version_2_with_negative = cfg.dataset.version_2_with_negative
        self.doc_stride = cfg.dataset.doc_stride
        self.max_query_length = cfg.dataset.max_query_length
        self.max_seq_length = cfg.dataset.max_seq_length
        self.do_lower_case = cfg.dataset.do_lower_case
        self.use_cache = cfg.dataset.use_cache
        self.tokenizer = get_tokenizer(
            tokenizer_name=cfg.language_model.tokenizer,
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            vocab_file=cfg.language_model.vocab_file,
            tokenizer_model=cfg.language_model.tokenizer_model,
            do_lower_case=cfg.dataset.do_lower_case,
        )

        super().__init__(cfg=cfg, trainer=trainer)

        self.bert_model = get_pretrained_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            config_file=cfg.language_model.bert_config,
            checkpoint_file=cfg.language_model.bert_checkpoint,
        )

        self.hidden_size = self.bert_model.hidden_size
        self.classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=cfg.token_classifier.num_classes,
            num_layers=cfg.token_classifier.num_layers,
            activation=cfg.token_classifier.activation,
            log_softmax=cfg.token_classifier.log_softmax,
            dropout=cfg.token_classifier.dropout,
            use_transformer_init=cfg.token_classifier.use_transformer_init,
        )

        self.loss = SpanningLoss()

    @typecheck()
    def forward(self, input_ids, token_type_ids, attention_mask):
        hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        logits = self.classifier(hidden_states=hidden_states)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, input_type_ids, input_mask, unique_ids, start_positions, end_positions = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        loss, _, _ = self.loss(logits=logits, start_positions=start_positions, end_positions=end_positions)

        tensorboard_logs = {'train_loss': loss, 'lr': self._optimizer.param_groups[0]['lr']}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_ids, input_type_ids, input_mask, unique_ids, start_positions, end_positions = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        loss, start_logits, end_logits = self.loss(
            logits=logits, start_positions=start_positions, end_positions=end_positions
        )

        eval_tensors = {
            'unique_ids': unique_ids,
            'start_logits': start_logits,
            'end_logits': end_logits,
        }
        return {'val_loss': loss, 'eval_tensors': eval_tensors}

    def test_step(self, batch, batch_idx):
        input_ids, input_type_ids, input_mask, unique_ids = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        test_tensors = {
            'unique_ids': unique_ids,
            'logits': logits,
        }
        return {'test_tensors': test_tensors}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        unique_ids = torch.cat([x['eval_tensors']['unique_ids'] for x in outputs])
        start_logits = torch.cat([x['eval_tensors']['start_logits'] for x in outputs])
        end_logits = torch.cat([x['eval_tensors']['end_logits'] for x in outputs])

        all_unique_ids = []
        all_start_logits = []
        all_end_logits = []
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            for ind in range(world_size):
                all_unique_ids.append(torch.empty_like(unique_ids))
                all_start_logits.append(torch.empty_like(start_logits))
                all_end_logits.append(torch.empty_like(end_logits))
            torch.distributed.all_gather(all_unique_ids, unique_ids)
            torch.distributed.all_gather(all_start_logits, start_logits)
            torch.distributed.all_gather(all_end_logits, end_logits)
        else:
            all_unique_ids.append(unique_ids)
            all_start_logits.append(start_logits)
            all_end_logits.append(end_logits)

        exact_match, f1, all_predictions, all_nbest = -1, -1, [], []
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:

            unique_ids = []
            start_logits = []
            end_logits = []
            for u in all_unique_ids:
                unique_ids.extend(u.cpu().numpy().tolist())
            for u in all_start_logits:
                start_logits.extend(u.cpu().numpy().tolist())
            for u in all_end_logits:
                end_logits.extend(u.cpu().numpy().tolist())

            exact_match, f1, all_predictions, all_nbest = self.validation_dataset.evaluate(
                unique_ids=unique_ids,
                start_logits=start_logits,
                end_logits=end_logits,
                n_best_size=self.validation_config.n_best_size,
                max_answer_length=self.validation_config.max_answer_length,
                version_2_with_negative=self.version_2_with_negative,
                null_score_diff_threshold=self.validation_config.null_score_diff_threshold,
                do_lower_case=self.do_lower_case,
            )

            if self.validation_config.output_nbest_file is not None:
                with open(self.validation_config.output_nbest_file, "w") as writer:
                    writer.write(json.dumps(all_nbest, indent=4) + "\n")
            if self.validation_config.output_prediction_file is not None:
                with open(self.validation_config.output_prediction_file, "w") as writer:
                    writer.write(json.dumps(all_predictions, indent=4) + "\n")

        logging.info(f"exact match {exact_match}")
        logging.info(f"f1 {f1}")

        tensorboard_logs = {'val_loss': avg_loss, 'exact_match': exact_match, 'f1': f1}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):

        unique_ids = torch.cat([x['test_tensors']['unique_ids'] for x in outputs])
        logits = torch.cat([x['test_tensors']['logits'] for x in outputs])

        all_unique_ids = []
        all_logits = []
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            for ind in range(world_size):
                all_unique_ids.append(torch.empty_like(unique_ids))
                all_logits.append(torch.empty_like(logits))
            torch.distributed.all_gather(all_unique_ids, unique_ids)
            torch.distributed.all_gather(all_logits, logits)
        else:
            all_unique_ids.append(unique_ids)
            all_logits.append(logits)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:

            unique_ids = []
            start_logits = []
            end_logits = []
            for u in all_unique_ids:
                unique_ids.extend(u.cpu().numpy().tolist())
            for u in all_logits:
                s, e = u.split(dim=-1, split_size=1)
                start_logits.extend(s.squeeze().cpu().numpy().tolist())
                end_logits.extend(e.squeeze().cpu().numpy().tolist())

            (all_predictions, all_nbest, scores_diff) = self.test_dataset.get_predictions(
                unique_ids=unique_ids,
                start_logits=start_logits,
                end_logits=end_logits,
                n_best_size=self.test_config.n_best_size,
                max_answer_length=self.test_config.max_answer_length,
                version_2_with_negative=self.version_2_with_negative,
                null_score_diff_threshold=self.test_config.null_score_diff_threshold,
                do_lower_case=self.do_lower_case,
            )

            if self.test_config.output_nbest_file is not None:
                with open(self.test_config.output_nbest_file, "w") as writer:
                    writer.write(json.dumps(all_nbest, indent=4) + "\n")
            if self.test_config.output_prediction_file is not None:
                with open(self.test_config.output_prediction_file, "w") as writer:
                    writer.write(json.dumps(all_predictions, indent=4) + "\n")

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)
        self.validation_config = val_data_config

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)
        self.test_config = test_data_config

    def _setup_dataloader_from_config(self, cfg: DictConfig):
        dataset = SquadDataset(
            tokenizer=self.tokenizer,
            data_file=cfg.file,
            doc_stride=self.doc_stride,
            max_query_length=self.max_query_length,
            max_seq_length=self.max_seq_length,
            version_2_with_negative=self.version_2_with_negative,
            num_samples=cfg.num_samples,
            mode=cfg.mode,
            use_cache=self.use_cache,
        )
        if cfg.mode == "eval":
            self.validation_dataset = dataset
        elif cfg.mode == "test":
            self.test_dataset = dataset

        dl = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            collate_fn=dataset.collate_fn,
            drop_last=cfg.get('drop_last', False),
            shuffle=cfg.shuffle,
            num_workers=cfg.get('num_workers', 0),
        )
        return dl

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass
