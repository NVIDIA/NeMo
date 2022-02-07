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

import os
import json
import pickle
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.t0.multitask_data_manager import get_guid
from nemo.collections.nlp.data.language_modeling.megatron.t5_dataset import (
    make_attention_mask_3d,
    make_history_mask_3d,
)
from nemo.core.classes import Dataset
from nemo.core.neural_types import CategoricalValuesType, ChannelType, MaskType, NeuralType, RegressionValuesType
from nemo.utils import logging


class InputPromptedExample(object):
    """A single training/test example for prompted inputs.

    Args:
        guid: Unique id for the example.
        text: The untokenized text of the first sequence.
        For single sequence tasks, only this sequence must be specified.
        prompt_type: Name of prompt applied to the exampled.
        label:The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid: int, text: str, prompt_id: int = None, label: str = None):
        """Constructs a InputExample."""
        self.guid = guid
        self.input_text = text
        self.prompt_id = prompt_id
        self.label = label

    def __repr__(self):
        return (
            f"InputExample(guid='{self.guid}', input_text='{self.input_text}', "
            f"prompt_type='{self.prompt_type}', label='{self.label}')"
        )



class T0Dataset(Dataset):
    """T0 Dataset in a text-to-text format."""

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return

    def __init__(
            self,
            file_path: str,
            task_name: str,
            subset: str,
            tokenizer: TokenizerSpec,
            max_seq_length: int,
            max_seq_length_decoder: int = 128,
            use_cache: bool = True,
            prefix_override: str = None,
            extension: str = 'jsonl'
    ):
        """
        Processes T0 dataset
        Args:
            file_name: path to file
            task_name: T0 task name
            tokenizer: such as AutoTokenizer
            max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
            use_cache: whether to use data cache
            prefix_override: if you want to override default prompt for this task specify this via a string.
        """
        logging.info(f'Processing {file_path}')

        self.tokenizer = tokenizer
        self.use_cache = use_cache
        self.extension = extension
        self.prompt_type_id = {}
        evaluate = False if 'train' in file_path else True  # TODO: needed?

        self.max_seq_length = max_seq_length
        self.max_seq_length_decoder = max_seq_length_decoder
        self.prefix_override = prefix_override
        #TODO: add non-prompted data get_example method
        self.examples = self.get_prompted_examples(file_path, task_name, subset)
        self.features = self.get_features(file_path)

    def get_features(self, file_path):
        data_dir, file_name = os.path.split(file_path)
        file_name = file_name.split('.%s' % self.extension)[0]
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}".format(
                file_name, self.tokenizer.name,
                str(self.max_seq_length ), str(getattr(self.tokenizer, "vocab_size", 0))
            ),
        )

        if self.use_cache and os.path.exists(cached_features_file):
            logging.info(f"loading from {cached_features_file}")
            with open(cached_features_file, "rb") as reader:
                features = pickle.load(reader)
        else:
            features = self.convert_examples_to_features()
            master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            if master_device:
                logging.info(f'Saving train features into {cached_features_file}')
                with open(cached_features_file, "wb") as writer:
                    pickle.dump(self.features, writer)
        return features

    def get_prompted_examples(self, file_path, task_name, subset):
        examples = []
        with open(file_path, "r") as f:
            for row in f:
                multi_prompted_ex = json.loads(row)
                for prompt_type, data in multi_prompted_ex.items():
                    self.prompt_id[prompt_type] = self.prompt_id.get(prompt_type, len(self.prompt_id)+1)
                    examples.append(InputPromptedExample(
                        guid=get_guid(task_name, subset),
                        text=data['input'],
                        prompt_id=self.prompt_id[prompt_type],
                        label=data['label']
                    ))
        return examples

    def convert_examples_to_features(self):
        """
        Converts examples into Text-to-Text batches to be used with a model like T5.
        Inputs are prefixed with a text prompt that indicates the task to perform.
        """
        features = []
        for ex_index, example in enumerate(self.examples):
            if ex_index % 10000 == 0:
                logging.info(f"Writing example {ex_index} of {len(self.examples)}")

            enc_query = self.tokenizer.text_to_ids(example.input_text)
            if len(enc_query) > self.max_seq_length:
                enc_query = enc_query[: self.max_seq_length]
            dec_query = (
                    [self.tokenizer.cls_id]
                    + self.tokenizer.text_to_ids(example.label)
                    + [self.tokenizer.eos_id]
            )

            dec_input = dec_query[:-1]
            labels = dec_query[1:]

            features.append([enc_query, dec_input, labels, example.guid, example.prompt_id])

        return features

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        enc_query, dec_input, labels, guid, prompt_id = self.features[idx]
        return {
            'text_enc': enc_query,
            'text_dec': dec_input,
            'labels': labels,
            'guid': guid,
            'prompt_id': prompt_id
        }

    def collate_fn(self, batch):
        enc_query = [item['text_enc'] for item in batch]
        dec_input = [item['text_dec'] for item in batch]
        labels = [item['labels'] for item in batch]
        guids = [item['guid'] for item in batch]
        prompt_ids = [item['prompt_id'] for item in batch]

        max_dec_input_length = max([len(item) for item in dec_input])
        max_enc_query_length = max([len(item) for item in enc_query])
        max_label_length = max([len(item) for item in labels])

        loss_mask = [([1] * (len(item))) + ([0] * (max_label_length - len(item))) for item in labels]
        enc_query = [item + [self.tokenizer.pad_id] * (max_enc_query_length - len(item)) for item in enc_query]
        dec_input = [item + [self.tokenizer.pad_id] * (max_dec_input_length - len(item)) for item in dec_input]
        labels = [item + [self.tokenizer.pad_id] * (max_label_length - len(item)) for item in labels]

        enc_query = torch.LongTensor(enc_query)
        dec_input = torch.LongTensor(dec_input)
        labels = torch.LongTensor(labels)
        loss_mask = torch.LongTensor(loss_mask)

        enc_mask = make_attention_mask_3d(enc_query, enc_query, self.tokenizer.pad_id).long()
        dec_mask = make_attention_mask_3d(dec_input, dec_input, self.tokenizer.pad_id)
        dec_mask = (dec_mask * make_history_mask_3d(dec_input)).long()
        enc_dec_mask = make_attention_mask_3d(dec_input, enc_query, self.tokenizer.pad_id).long()

        return {
            'text_enc': enc_query,
            'text_dec': dec_input,
            'labels': labels,
            'loss_mask': loss_mask,
            'enc_mask': enc_mask,
            'dec_mask': dec_mask,
            'enc_dec_mask': enc_dec_mask,
            'guids': guids,
            'prompt_ids': prompt_ids
        }

    def make_history_mask_3d(self, block):
        batch, length = block.shape
        arange = np.arange(length)
        history_mask = (arange[None,] <= arange[:, None])[
            None,
        ]
        history_mask = np.repeat(history_mask, batch, 0)
        return history_mask


