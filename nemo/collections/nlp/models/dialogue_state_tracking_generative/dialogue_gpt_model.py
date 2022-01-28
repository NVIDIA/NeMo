# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2019 The Google Research Authors.
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

'''
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/google-research/blob/master/schema_guided_dst/baseline/train_and_predict.py
'''

import collections
import copy
import os
import re
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import functional as F
from torch import nn
from torch.utils.data import DataLoader

from nemo.collections.nlp.data.dialogue_state_tracking.sgd.evaluate import evaluate, get_in_domain_services
from nemo.collections.nlp.data.dialogue_state_tracking.sgd.prediction_utils import write_predictions_to_file
from nemo.collections.nlp.data.dialogue_state_tracking_generative import (
    DialogueGPTDataset,
    DialogueSGDBERTDataset,
    DialogueSGDDataProcessor,
    Schema,
)
from nemo.collections.nlp.losses import SGDDialogueStateLoss
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules import SGDDecoder, SGDEncoder
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import NeuralType
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero

__all__ = ['DialogueGPTModel']

NUM_TASKS = 1  # focussing on intent currently 6  # number of multi-head tasks


class DialogueGPTModel(NLPModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.data_prepared = False

        self.setup_tokenizer(cfg.tokenizer)
        super().__init__(cfg=cfg, trainer=trainer)

        self.language_model = get_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            config_file=self.register_artifact('language_model.config_file', cfg.language_model.config_file),
            config_dict=OmegaConf.to_container(cfg.language_model.config) if cfg.language_model.config else None,
            checkpoint_file=cfg.language_model.lm_checkpoint,
            vocab_file=self.register_artifact('tokenizer.vocab_file', cfg.tokenizer.vocab_file),
        )

        all_labels = list(self._train_dl.dataset.all_possible_labels)

        self.label_to_ids = collections.defaultdict(int)
        # 0 is reserved for unseen labels
        for i in range(len(all_labels)):
            self.label_to_ids[all_labels[i]] = i + 1

        self.language_model.resize_token_embeddings(len(self.tokenizer.tokenizer))
        self.token_to_words = {}
        self.classification_report = ClassificationReport(
            num_classes=len(self.label_to_ids) + 1, mode='micro', label_ids=self.label_to_ids, dist_sync_on_step=True
        )
        self.eval_mode = cfg.eval_mode

    def training_step(self, batch, batch_idx):
        (
            input_ids,
            attn_masks,
            labels,
            generate_input_ids,
            generate_attn_masks,
            candidate_input_ids,
            candidate_attn_masks,
            template_length,
            utterance_length,
        ) = batch
        loss, logits = self.language_model(input_ids=input_ids, attention_mask=attn_masks, labels=labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'logits': logits}

    def validation_step(self, batch, batch_idx):
        return self.eval_step_helper(batch=batch)

    def validation_epoch_end(self, outputs):
        self.eval_epoch_end(outputs)

    def test_epoch_end(self, outputs):
        self.eval_epoch_end(outputs)

    def eval_epoch_end(self, outputs):
        precision, recall, f1, report = self.classification_report.compute()

        logging.info(report)
        acc = np.mean([output["acc"] for output in outputs])

        self.log('precision', precision)
        self.log('f1', f1)
        self.log('recall', recall)
        self.log('accuracy', acc * 100)

    def test_step(self, batch, batch_idx):
        return self.eval_step_helper(batch=batch, mode='test')

    # for inference only
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        raise NotImplementedError()
        return self.model(batch)

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        loss, logits = self.language_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return loss, logits

    def decode(self, tokens):
        if tokens not in self.token_to_words:
            self.token_to_words[tokens] = self.tokenizer.tokenizer.decode(tokens)
        return self.token_to_words[tokens]

    def rank_candidates(
        self, candidate_input_ids, candidate_attn_masks, utterance_length, labels, template_length, minus_prior=False
    ):
        best_candidate_input_ids = []

        for i in range(candidate_input_ids.size(0)):
            best_j = 0

            lowest_loss = float("inf")

            utterance_end = utterance_length[i].item()
            for j in range(candidate_input_ids.size(1)):

                if 0 < j < candidate_input_ids.size(1) and torch.equal(
                    candidate_input_ids[i, j, :], candidate_input_ids[i, 0, :]
                ):
                    break

                cand_loss, _ = self.language_model(
                    input_ids=candidate_input_ids[i, j : j + 1, :],
                    attention_mask=candidate_attn_masks[i, j : j + 1, :],
                    labels=candidate_input_ids[i, j : j + 1, :],
                )

                considered_loss = cand_loss.item()

                if minus_prior:
                    utterance_free_cand_loss, _ = self.language_model(
                        input_ids=candidate_input_ids[i, j : j + 1, utterance_end:],
                        attention_mask=candidate_attn_masks[i, j : j + 1, utterance_end:],
                        labels=candidate_input_ids[i, j : j + 1, utterance_end:],
                    )
                    considered_loss -= utterance_free_cand_loss.item()

                if considered_loss < lowest_loss:
                    best_j = j
                    lowest_loss = considered_loss

            best_candidate_input_ids.append(candidate_input_ids[i, best_j, :])

        candidate_tokens = torch.stack(best_candidate_input_ids)
        generated_field, ground_truth_field = self.process_into_structured_fields(
            candidate_tokens, labels, left_padding=False, template_length=template_length
        )
        return generated_field, ground_truth_field

    def generate_candidates(self, generate_input_ids, generate_attn_masks, labels):
        param_dict = {
            "input_ids": generate_input_ids,
            "attention_masks": generate_attn_masks,
            "max_length": self._cfg.dataset.max_seq_length + 32,
            "pad_token_id": self.tokenizer.tokenizer.pad_token_id,
        }

        generated_tokens = self.language_model.generate(**param_dict)
        generated_field, ground_truth_field = self.process_into_structured_fields(generated_tokens, labels)

        return generated_field, ground_truth_field

    def eval_step_helper(self, batch, mode='val'):
        (
            input_ids,
            attn_masks,
            labels,
            generate_input_ids,
            generate_attn_masks,
            candidate_input_ids,
            candidate_attn_masks,
            template_length,
            utterance_length,
        ) = batch

        loss, logits = self.language_model(input_ids=input_ids, attention_mask=attn_masks, labels=labels)

        self.log("{}_loss".format(mode), loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # ranking using perplexity of candidates
        if self.eval_mode == "ranking":
            generated_field, ground_truth_field = self.rank_candidates(
                candidate_input_ids, candidate_attn_masks, utterance_length, labels, template_length
            )
        # generate candidates (possibly with constraint)
        elif self.eval_mode == "generation":
            generated_field, ground_truth_field = self.generate_candidates(
                generate_input_ids, generate_attn_masks, labels
            )
        else:
            raise ValueError("{} is not among supported options (ranking, generation)".format(self.eval_mode))
        generated_field_ids = torch.tensor([self.label_to_ids[label] for label in generated_field], dtype=int).to(
            labels.device
        )
        ground_truth_field_ids = torch.tensor(
            [self.label_to_ids[label] for label in ground_truth_field], dtype=int
        ).to(labels.device)

        tp, fn, fp, _ = self.classification_report(generated_field_ids, ground_truth_field_ids)
        acc = np.mean(
            [int(generated_field[i].strip() == ground_truth_field[i].strip()) for i in range(len(generated_field))]
        )

        return {
            'loss': loss,
            'generated_field': generated_field,
            'ground_truth_field': ground_truth_field,
            'tp': tp,
            'fn': fn,
            'fp': fp,
            'acc': acc,
        }

    def process_into_structured_fields(self, generated_tokens, labels, left_padding=True, template_length=None):

        generated_field = []

        for i in range(generated_tokens.size(0)):
            if left_padding and template_length is None:
                start_point = self._cfg.dataset.max_seq_length
            else:
                start_point = template_length[i].item()

            stop_point = generated_tokens.size(1)

            for j in range(start_point, stop_point):
                if generated_tokens.data[i, j] == self.tokenizer.tokenizer.pad_token_id:
                    stop_point = j
                    break
            generated_field.append(self.decode(generated_tokens[i, start_point:stop_point]).strip())

        ground_truth_field = []

        for i in range(labels.size(0)):
            correct_label = tuple(
                [j for j in labels.data[i] if j != self.tokenizer.tokenizer.pad_token_id and j != -100]
            )
            ground_truth_field.append(self.decode(correct_label).strip())

        return generated_field, ground_truth_field

    def prepare_data(self):
        """
        Preprocessed schema and dialogues and caches this
        """
        if self.data_prepared:
            return
        schema_config = {
            "MAX_NUM_CAT_SLOT": self._cfg.dataset.max_num_cat_slot,
            "MAX_NUM_NONCAT_SLOT": self._cfg.dataset.max_num_noncat_slot,
            "MAX_NUM_VALUE_PER_CAT_SLOT": self._cfg.dataset.max_value_per_cat_slot,
            "MAX_NUM_INTENT": self._cfg.dataset.max_num_intent,
            "NUM_TASKS": NUM_TASKS,
            "MAX_SEQ_LENGTH": self._cfg.dataset.max_seq_length,
        }
        all_schema_json_paths = []
        for dataset_split in ['train', 'test', 'dev']:
            all_schema_json_paths.append(os.path.join(self._cfg.dataset.data_dir, dataset_split, "schema.json"))
        schemas = Schema(all_schema_json_paths)

        self.dialogues_processor = DialogueSGDDataProcessor(
            task_name=self._cfg.dataset.task_name,
            data_dir=self._cfg.dataset.data_dir,
            dialogues_example_dir=self._cfg.dataset.dialogues_example_dir,
            tokenizer=self.tokenizer,
            schemas=schemas,
            schema_config=schema_config,
            subsample=self._cfg.dataset.subsample,
        )

        if is_global_rank_zero():
            overwrite_dial_files = not self._cfg.dataset.use_cache
            self.dialogues_processor.save_dialog_examples(overwrite_dial_files=overwrite_dial_files)

        self.data_prepared = True

    def update_data_dirs(self, data_dir: str, dialogues_example_dir: str):
        """
        Update data directories

        Args:
            data_dir: path to data directory
            dialogues_example_dir: path to preprocessed dialogues example directory, if not exists will be created.
        """
        if not os.path.exists(data_dir):
            raise ValueError(f"{data_dir} is not found")
        self._cfg.dataset.data_dir = data_dir
        self._cfg.dataset.dialogues_example_dir = dialogues_example_dir
        logging.info(f'Setting model.dataset.data_dir to {data_dir}.')
        logging.info(f'Setting model.dataset.dialogues_example_dir to {dialogues_example_dir}.')

    def setup_training_data(self, train_data_config: Optional[DictConfig] = None):
        self.prepare_data()
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, split=train_data_config.ds_item)

    def setup_validation_data(self, val_data_config: Optional[DictConfig] = None):
        self.prepare_data()
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, split=val_data_config.ds_item)

    def setup_test_data(self, test_data_config: Optional[DictConfig] = None):
        self.prepare_data()
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, split=test_data_config.ds_item)

    def _setup_dataloader_from_config(self, cfg: DictConfig, split: str) -> DataLoader:
        dataset_cfg = self._cfg.dataset
        data_dir = dataset_cfg.data_dir

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory is not found at: {data_dir}.")
        if dataset_cfg.task == 'sgd':
            dataset = DialogueGPTDataset(
                dataset_split=split,
                dialogues_processor=self.dialogues_processor,
                tokenizer=self.dialogues_processor._tokenizer,
                cfg=dataset_cfg,
            )
        else:
            raise NotImplementedError("Task {} has not been implemented".format(dataset_cfg.task_name))

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
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        return result
