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

"""
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/google-research/blob/master/schema_guided_dst
"""

import copy
import re
from collections import defaultdict

import torch

from nemo.core.classes import Dataset


class DialogueGPTDataset(Dataset):
    '''
    Dataset Class 
        1. Performs Model-dependent (but Data-independent) operations (tokenization etc)
        2. This can allow the same model preprocessing for multiple datasources
        3. Users can configurate which labels to use for modelling 
            (e.g. intent classification, slot filling or both together etc)
        
    '''

    def __init__(self, dataset_split: str, dialogues_processor: object, tokenizer, cfg):
        """ Constructor
        Args:
            dataset_split: dataset split
            dialogues_processor: Data generator for SGD dialogues
        """
        self.label_type = "intent"
        self.all_possible_labels = set()
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.max_candidates = 2
        self.label_to_description = defaultdict(str)
        if not isinstance(dataset_split, str):
            dataset_split = dataset_split[0]
        self.features = dialogues_processor.get_dialog_examples(dataset_split)
        for idx in range(len(self.features)):
            self.preprocess_feature(idx)

    def transform(self, label):
        if self.cfg.task == "sgd":
            label = self.convert_camelcase_to_lower(label)
        elif self.cfg.task == "assistant":
            label = label.replace('_', ' ')
        return label

    def convert_camelcase_to_lower(self, label):
        if label.lower() == "none":
            return "none"
        label = label.split("_")[0]
        tokens = re.findall('[A-Z][^A-Z]*', label)
        return ' '.join([token.lower() for token in tokens])

    def __len__(self):
        return len(self.features)

    def get_n_tokens_in_sentence(self, sentence):
        encodings_dict = self.tokenizer.tokenizer(
            sentence, truncation=True, max_length=self.cfg.max_seq_length, padding=False, return_tensors="pt"
        )
        output = torch.squeeze(encodings_dict['input_ids'])
        return len(output) if len(output.size()) > 0 else 0

    def preprocess_feature(self, idx):
        ex = self.features[idx].data
        label = ex["labels"][self.label_type]
        candidates = ex["possible_labels"][self.label_type]

        if self.label_type in ["service", "intent"]:
            label = self.transform(label)
            candidates = [self.transform(candidate) for candidate in candidates]

        self.features[idx].data["labels"][self.label_type] = label
        self.features[idx].data["possible_labels"][self.label_type] = candidates
        # description = ex["description"][self.label_type]
        # self.label_to_description[label] = description

        for candidate in candidates:
            self.all_possible_labels.add(candidate)
        self.max_candidates = max(self.max_candidates, len(candidates) * 2)

    def default_encode(self, sentence):
        encodings_dict = self.tokenizer.tokenizer(
            sentence, truncation=True, max_length=self.cfg.max_seq_length, padding="max_length", return_tensors="pt"
        )
        input_ids = torch.squeeze(encodings_dict['input_ids'])
        attn_masks = torch.squeeze(encodings_dict['attention_mask'])
        return encodings_dict, input_ids, attn_masks

    def __getitem__(self, idx: int):

        '''
        State how the input and output samples look like

        This template can be changed

        Training example: 
            e.g. <utterance> service: restaurant
            e.g. <task description> <utterance> detected service: restaurant

        Inference example:
            e.g. <utterance> service: 
        '''
        ex = self.features[idx].data

        utterance = ex["utterance"]
        label = ex["labels"][self.label_type]
        candidates = ex["possible_labels"][self.label_type]

        base_template = utterance + ' ' + self.label_type + ':'
        # base_template = utterance + ' ' + 'I want to'

        utterance_length = self.get_n_tokens_in_sentence(utterance)

        sentence_without_answer = base_template

        sentence = base_template + ' ' + label  # + ' (' + self.label_to_description[label] + ')'

        if self.cfg.eval_mode == "binary_score":
            candidate_sentences = []
            for candidate in candidates:
                positive_answer = base_template + ' ' + candidate + ' Answer: ' + 'yes'
                negative_answer = base_template + ' ' + candidate + ' Answer: ' + 'no'
                if candidate == label:
                    correct_candidate = len(candidate_sentences) // 2
                    candidate_sentences.append(positive_answer)
                    candidate_sentences.append(negative_answer)
                else:
                    candidate_sentences.append(negative_answer)
                    candidate_sentences.append(positive_answer)
        else:
            correct_candidate = 0
            candidate_sentences = [
                base_template + ' ' + candidate  # + ' (' + self.label_to_description[candidate] + ')'
                for candidate in candidates
            ]

        self.tokenizer.tokenizer.padding_side = "right"

        encodings_dict, input_ids, attn_masks = self.default_encode(sentence)

        candidate_tokenized_sentences = [
            self.default_encode(candidate_sentence) for candidate_sentence in candidate_sentences
        ]
        while len(candidate_tokenized_sentences) < self.max_candidates:
            candidate_tokenized_sentences.append(candidate_tokenized_sentences[0])

        candidate_input_ids = torch.stack([i[1] for i in candidate_tokenized_sentences])
        candidate_attn_masks = torch.stack([i[2] for i in candidate_tokenized_sentences])

        labels = copy.copy(torch.squeeze(encodings_dict['input_ids']))

        training_mask_end = self.get_n_tokens_in_sentence(sentence_without_answer)

        labels.data = torch.tensor(
            [-100 if i < training_mask_end else labels.data[i] for i in range(len(labels.data))]
        )

        # left padding is for batch generation but right padding is essential for teacher force training
        self.tokenizer.tokenizer.padding_side = "left"

        encodings_dict_without_answer = self.tokenizer.tokenizer(
            sentence_without_answer,
            truncation=True,
            max_length=self.cfg.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        generate_input_ids = torch.squeeze(encodings_dict_without_answer['input_ids'])
        generate_attn_masks = torch.squeeze(encodings_dict_without_answer['attention_mask'])
        return (
            input_ids,
            attn_masks,
            labels,
            generate_input_ids,
            generate_attn_masks,
            candidate_input_ids,
            candidate_attn_masks,
            training_mask_end,
            utterance_length,
            correct_candidate,
        )
