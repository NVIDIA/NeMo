# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
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

# Some code of this file was adapted from the HuggingFace library available at
# https://github.com/huggingface/transformers

from typing import Dict, List, Optional, Union

import numpy as np
import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.t5_dataset import (
    make_attention_mask_3d,
    make_history_mask_3d,
)
from nemo.core.classes import Dataset
from nemo.core.neural_types import CategoricalValuesType, ChannelType, MaskType, NeuralType, RegressionValuesType
from nemo.utils import logging
import csv
import json

__all__ = ['GPTPTuneDataset']

SMALL_NUM = -100

class InputExample(object):
    """A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: The untokenized text of the first sequence.
        For single sequence tasks, only this sequence must be specified.
        text_b: The untokenized text of the second
        sequence. Only must be specified for sequence pair tasks.
        label:The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid: int, text_a: str, text_b: str = None, label: str = None):
        """Constructs a InputExample."""
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return (
            f"InputExample(guid='{self.guid}', text_a='{self.text_a}', text_b='{self.text_b}', label='{self.label}')"
        )


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, data_type: str, task_type: str):
        self.data_type = data_type
        self.task_type = task_type

    def get_examples(self, data_path):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_task_type(self):
        return self.task_type

    def get_ptune_query(self, text_a: str, text_b: str, prompt_token_id: int, max_seq_len: int ,templates: List[int], tokenizer: TokenizerSpec):
        raise NotImplemented()
 
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                # if sys.version_info[0] == 2:
                #     line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_jsonl(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            lines = []
            for line in f:
                lines.append(json.loads(line))
            return lines


class BoolQProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def __init__(self, data_type: str, task_type: str):
        super().__init__(data_type, task_type)

    def get_examples(self, data_path):
        """See base class."""
        return self._create_examples(self._read_jsonl(data_path), self.data_type)

    def get_labels(self):
        """See base class."""
        return ["yes", "no"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = line['prompt_tag']
            text_a = line['sentence']
            text_b = line['question']
            label = line['label']
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_ptune_query(self,
                        text_a: str,
                        text_b: str,
                        prompt_token_id: int,
                        max_seq_len: int,
                        templates: List[int],
                        tokenizer: TokenizerSpec):
        sentence_a = f" Paragraph: {text_a}"
        sentence_b = f" Question: {text_b}?"
        a_input_token_ids = tokenizer.text_to_ids(sentence_a)
        b_input_token_ids = tokenizer.text_to_ids(sentence_b)
        c_input_token_ids = tokenizer.text_to_ids(" Answer:")
        cut = 0
        total_num_ids = len(a_input_token_ids) + len(b_input_token_ids) + len(c_input_token_ids) + sum(templates) 
        if total_num_ids > max_seq_len:
            logging.warning("Input sequence is longer than the LM model max seq, will cut it off to fit")
            cut = total_num_ids - max_seq_len
        return [prompt_token_id] * templates[0] + a_input_token_ids[cut:] + \
               [prompt_token_id] * templates[1] + b_input_token_ids +  \
               [prompt_token_id] * templates[2] + c_input_token_ids

    def label2string(self, label):
        return ' '+label


processors = {
    "boolq": BoolQProcessor
}


class TaskDataset(Dataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            "labels": NeuralType(
                tuple('B'), RegressionValuesType() if self.task_name == 'sts-b' else CategoricalValuesType()
            ),
        }

    def __init__(
        self,
        file_name: str,
        task_name: str,
        data_type: str,
        tokenizer: TokenizerSpec,
    ):
        """
        Processes Task datasets
        Args:
            file_name: path to file
            task_name: task name
            tokenizer: such as AutoTokenizer
            max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
            use_cache: whether to use data cache
        """
        logging.info(f'Processing {file_name}')
        self.tokenizer = tokenizer
        if task_name not in processors:
            raise ValueError(f'{task_name} not supported. Choose from {processors.keys()}')

        self.processor = processors[task_name](data_type, task_name)
        self.label_list = self.processor.get_labels()
        self.examples = self.processor.get_examples(file_name)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        return (
            np.array(feature.input_ids),
            np.array(feature.segment_ids),
            np.array(feature.input_mask, dtype=np.long),
            np.array(feature.label_id),
        )


class GPTPTuneDataset(TaskDataset):
    """Multiple Task Dataset in a text-to-text format."""

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return

    def __init__(
        self,
        file_name: str,
        task_name: str,
        data_type: str,
        tokenizer: TokenizerSpec,
        templates: List[int],
        pseudo_token_id: int,
        pad_id: int,
        max_seq_length: int,
        max_seq_length_decoder: int = 128,
    ):
        """
        Processes TextToText PTuning Dataset
        Args:
            file_name: path to file
            task_name: nlp task name
            data_type: train/dev/test
            tokenizer: such as AutoTokenizer
            templates: virtual token template, list of integers
            max_seq_length: max sequence length for encoder
            max_seq_length_decoder: max seq length for decoder
        """
        super().__init__(file_name, task_name, data_type, tokenizer)
        self.max_seq_length = max_seq_length
        self.max_seq_length_decoder = max_seq_length_decoder
        self.templates = templates
        self.pseudo_token_id = pseudo_token_id
        self.pad_id = pad_id
        self.features = self.convert_examples_to_features()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids, labels, query  = self.features[idx]
        return {'input_enc': input_ids, 'labels': labels, 'query_enc': query}

    def collate_fn(self, batch):
        enc_input = [item['input_enc'] for item in batch]
        labels = [item['labels'] for item in batch]
        enc_query = [item['query_enc'] for item in batch]

        max_input_length = max([len(item) for item in enc_input])
        max_label_length = max([len(item) for item in labels])
        max_query_length = max([len(item) for item in enc_query])

        loss_mask = [([1] * (len(item))) + ([0] * (max_label_length - len(item))) for item in labels]
        enc_input = [item + [self.pad_id] * (max_input_length - len(item)) for item in enc_input]
        labels = [item + [self.pad_id] * (max_label_length - len(item)) for item in labels]
        enc_query = [item + [self.pad_id] * (max_query_length - len(item)) for item in enc_query]

        enc_query = torch.LongTensor(enc_query)
        enc_input = torch.LongTensor(enc_input)
        labels = torch.LongTensor(labels)
        loss_mask = torch.LongTensor(loss_mask)
        loss_mask[labels == SMALL_NUM] = 0

        input_attn_mask = make_attention_mask_3d(enc_input, enc_input, self.pad_id)
        input_attn_mask = (input_attn_mask * make_history_mask_3d(enc_input)).long()

        return {
            'enc_input': enc_input,
            'labels': labels,
            'loss_mask': loss_mask,
            'enc_query': enc_query,
            'input_attn_mask': input_attn_mask,
        }

    def convert_examples_to_features(self):
        """
        Converts examples into Text-to-Text batches to be used with a model like T5.
        Inputs are prefixed with a text prompt that indicates the task to perform.
        """
        features = []
        for ex_index, example in enumerate(self.examples):
            if ex_index % 10000 == 0:
                logging.info(f"Writing example {ex_index} of {len(self.examples)}")

            label_ids = self.tokenizer.text_to_ids(self.processor.label2string(example.label)) + [self.tokenizer.eos_id]
            enc_query = self.processor.get_ptune_query(example.text_a, example.text_b, self.pseudo_token_id, self.max_seq_length - len(label_ids) + 1, self.templates, self.tokenizer)
            input_ids = enc_query + label_ids[:-1]
            labels = [SMALL_NUM for i in range(len(enc_query) - 1)] + label_ids
            features.append([input_ids, labels, enc_query])
        return features
