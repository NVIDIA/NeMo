# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import os
from typing import Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from transformers import AutoModelWithLMHead

from nemo.collections.nlp.data.dialogue_state_tracking_generative import (
    DialogueGPTDataset,
    DialogueSGDDataProcessor,
    Schema,
)
from nemo.collections.nlp.data.dialogue_state_tracking_generative.sgd.assistant_data_processor import (
    DialogueAssistantDataProcessor,
)
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero

__all__ = ['DialogueGPTModel']

NUM_TASKS = 1  # focussing on intent currently 6  # number of multi-head tasks


class DialogueGPTModel(NLPModel):
    def __init__(
        self, cfg: DictConfig, trainer: Trainer = None,
    ):
        self.data_prepared = False
        self.setup_tokenizer(cfg.tokenizer)
        super().__init__(cfg=cfg, trainer=trainer)
        if cfg.library == "huggingface":
            self.language_model = AutoModelWithLMHead.from_pretrained(cfg.language_model.pretrained_model_name)
            self.language_model.resize_token_embeddings(len(self.tokenizer.tokenizer))
        elif cfg.library == "megatron":
            self.language_model = MegatronGPTModel.restore_from(cfg.language_model.lm_checkpoint, trainer=trainer)

        all_labels = list(
            self._train_dl.dataset.all_possible_labels.union(
                self._validation_dl.dataset.all_possible_labels, self._test_dl.dataset.all_possible_labels
            )
        )
        self.label_to_ids = collections.defaultdict(int)

        for i in range(len(all_labels)):
            self.label_to_ids[all_labels[i]] = i

        self.all_existing_labels = set(self.label_to_ids.keys())

        self.token_to_words = {}
        self.classification_report = ClassificationReport(
            num_classes=len(self.label_to_ids) + 1, mode='micro', label_ids=self.label_to_ids, dist_sync_on_step=True
        )
        self.eval_mode = cfg.eval_mode
        self.cfg = cfg

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
            correct_candidate,
        ) = batch

        if self.eval_mode == "binary_score":
            new_input_ids = []
            new_attn_masks = []
            for i in range(candidate_input_ids.size(0)):
                for j in range(0, candidate_input_ids.size(1), 2):
                    if j > 0 and torch.equal(candidate_input_ids[i, j, :], candidate_input_ids[i, 0, :]):
                        break
                    new_input_ids.append(candidate_input_ids[i, j, :])
                    new_attn_masks.append(candidate_attn_masks[i, j, :])
            input_ids = torch.stack(new_input_ids)
            attn_masks = torch.stack(new_attn_masks)
            labels = self.get_binary_score_labels(input_ids)

        loss = self(input_ids, attn_masks, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        return self.eval_step_helper(batch=batch)

    def validation_epoch_end(self, outputs):
        self.eval_epoch_end(outputs, mode='val')

    def test_epoch_end(self, outputs):
        self.eval_epoch_end(outputs, mode='test')

    def eval_epoch_end(self, outputs, mode='val'):
        precision, recall, f1, report = self.classification_report.compute()

        logging.info(report)
        acc = np.mean([output["acc"] for output in outputs])

        self.log('precision', precision)
        self.log('f1', f1)
        self.log('recall', recall)
        self.log('{}_accuracy'.format(mode), acc * 100)

    def test_step(self, batch, batch_idx):
        return self.eval_step_helper(batch=batch, mode='test')

    # for inference only
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # return self(batch)
        raise NotImplementedError()

    def forward(self, input_ids, attention_mask, labels):

        if self.cfg.library == "huggingface":
            output = self.language_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output['loss']
        elif self.cfg.library == "megatron":
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).repeat(input_ids.size(0), 1)
            unmasked_unreduced_loss = self.language_model(
                input_ids, position_ids, attention_mask=attention_mask > 0, labels=labels
            )

            # labels_mask = torch.tensor([0 if (i == -100 or i == self.tokenizer.tokenizer.pad_token_id) else 1 for i in labels])
            filler = torch.zeros_like(labels)
            labels_mask_0 = torch.where(labels != -100, labels, filler)
            labels_mask_1 = torch.abs(torch.where(labels != self.tokenizer.tokenizer.pad_token_id, labels, filler))
            # labels_mask is where labels is neither -100 nor the pad token id
            labels_mask_with_id = torch.minimum(labels_mask_0, labels_mask_1)
            labels_mask = labels_mask_with_id > 0

            loss = self.language_model.loss_func(labels_mask, unmasked_unreduced_loss)
            loss = average_losses_across_data_parallel_group([loss])

        return loss

    def decode(self, tokens):
        if tokens not in self.token_to_words:
            self.token_to_words[tokens] = self.tokenizer.tokenizer.decode(tokens)
        return self.token_to_words[tokens]

    def binary_score_candidates(
        self,
        candidate_input_ids,
        candidate_attn_masks,
        utterance_length,
        labels,
        template_length,
        correct_candidate,
        minus_negative=True,
    ):
        best_candidate_input_ids = []

        for i in range(candidate_input_ids.size(0)):
            best_j = 0

            lowest_loss = float("inf")

            for j in range(0, candidate_input_ids.size(1), 2):

                if j > 0 and torch.equal(candidate_input_ids[i, j, :], candidate_input_ids[i, 0, :]):
                    break

                start_yes = j if j // 2 == correct_candidate[i].item() else j + 1

                cand_loss = self(
                    candidate_input_ids[i, start_yes : start_yes + 1, :],
                    candidate_attn_masks[i, start_yes : start_yes + 1, :],
                    self.get_binary_score_labels(candidate_input_ids[i, start_yes : start_yes + 1, :]),
                )

                considered_loss = cand_loss.item()

                if minus_negative:
                    start_no = j + 1 if j // 2 == correct_candidate[i].item() else j

                    negative_cand_loss = self(
                        candidate_input_ids[i, start_no : start_no + 1, :],
                        candidate_attn_masks[i, start_no : start_no + 1, :],
                        self.get_binary_score_labels(candidate_input_ids[i, start_no : start_no + 1, :]),
                    )
                    considered_loss -= negative_cand_loss.item()

                if considered_loss < lowest_loss:
                    best_j = start_yes
                    lowest_loss = considered_loss

            best_candidate_input_ids.append(candidate_input_ids[i, best_j, :])

        candidate_tokens = torch.stack(best_candidate_input_ids)
        generated_field, ground_truth_field = self.process_into_structured_fields(
            candidate_tokens, labels, left_padding=False, template_length=template_length
        )
        return generated_field, ground_truth_field

    def get_binary_score_labels(self, input_ids):
        # mask out every token except the last token for yes/no/true/false
        labels = torch.zeros_like(input_ids)

        for i in range(input_ids.size(0)):
            for j in range(input_ids.size(1)):
                if input_ids.data[0, j] == self.tokenizer.tokenizer.pad_token_id:
                    stop_point = j
                    break
            last_point = stop_point - 1
            labels.data[i, last_point] = input_ids[i, last_point]

        return labels

    def rank_candidates(
        self, candidate_input_ids, candidate_attn_masks, utterance_length, labels, template_length, minus_prior=True
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

                cand_loss = self(
                    candidate_input_ids[i, j : j + 1, :],
                    candidate_attn_masks[i, j : j + 1, :],
                    candidate_input_ids[i, j : j + 1, :],
                )

                considered_loss = cand_loss.item()

                if minus_prior:
                    utterance_free_cand_loss = self(
                        candidate_input_ids[i, j : j + 1, utterance_end:],
                        candidate_attn_masks[i, j : j + 1, utterance_end:],
                        candidate_input_ids[i, j : j + 1, utterance_end:],
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
        if self.cfg.library == "huggingface":
            param_dict = {
                "input_ids": generate_input_ids,
                "attention_masks": generate_attn_masks,
                "max_length": self._cfg.dataset.max_seq_length + 32,
                "pad_token_id": self.tokenizer.tokenizer.pad_token_id,
            }

            generated_tokens = self.language_model.generate(**param_dict)
        elif self.cfg.library == "megatron":
            raise NotImplementedError()
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
            correct_candidate,
        ) = batch

        loss = self(input_ids, attn_masks, labels)

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
        elif self.eval_mode == "binary_score":
            generated_field, ground_truth_field = self.binary_score_candidates(
                candidate_input_ids, candidate_attn_masks, utterance_length, labels, template_length, correct_candidate
            )

        else:
            raise ValueError(
                "{} is not among supported options (ranking, generation, binary_score)".format(self.eval_mode)
            )

        generated_field_ids = torch.tensor(
            [self.label_to_ids[label.strip()] for label in generated_field], dtype=int
        ).to(labels.device)
        ground_truth_field_ids = torch.tensor(
            [self.label_to_ids[label.strip()] for label in ground_truth_field], dtype=int
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
            # this is to account for the tokens ' Answer: ' + 'yes'/'no'/'true'/'false'
            if self.eval_mode == "binary_score":
                stop_point -= 3
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

        if self._cfg.dataset.task == 'sgd':
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
        elif self._cfg.dataset.task == 'assistant':
            self.dialogues_processor = DialogueAssistantDataProcessor(
                data_dir=self._cfg.dataset.data_dir, tokenizer=self.tokenizer,
            )

        if is_global_rank_zero():
            overwrite_dial_files = not self._cfg.dataset.use_cache
            if self._cfg.dataset.task == 'sgd':
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

    def setup_multiple_validation_data(self, val_data_config: Optional[DictConfig] = None):
        return self.setup_validation_data(val_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig] = None):
        self.prepare_data()
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, split=val_data_config.ds_item)

    def setup_multiple_test_data(self, test_data_config: Union[DictConfig, Dict]):
        self.setup_test_data(test_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig] = None):
        self.prepare_data()
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, split=test_data_config.ds_item)

    def _setup_dataloader_from_config(self, cfg: DictConfig, split: str) -> DataLoader:
        dataset_cfg = self._cfg.dataset
        data_dir = dataset_cfg.data_dir

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory is not found at: {data_dir}.")

        dataset = DialogueGPTDataset(
            dataset_split=split,
            dialogues_processor=self.dialogues_processor,
            tokenizer=self.dialogues_processor._tokenizer,
            cfg=dataset_cfg,
        )

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
