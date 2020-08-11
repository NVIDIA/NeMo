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

# TODO: which part is from HF?
"""
Utility functions for Token Classification NLP tasks
Some parts of this code were adapted from the HuggingFace library at
https://github.com/huggingface/pytorch-pretrained-BERT
"""

import os
import random
from typing import Any, Dict, Optional

import h5py
import numpy as np

from nemo.collections.nlp.data.data_utils.data_preprocessing import get_stats
from nemo.collections.nlp.parts.utils_funcs import list2str
from nemo.core.classes import Dataset
from nemo.core.neural_types import ChannelType, LabelsType, MaskType, NeuralType
from nemo.utils import logging

__all__ = ['TextClassificationDataset']


class TextClassificationDataset(Dataset):
    """A dataset class that converts from raw data to
    a dataset that can be used by DataLayerNM.
    Args:
        input_file: file to sequence + label.
            the first line is header (sentence [tab] label)
            each line should be [sentence][tab][label]
        tokenizer: tokenizer object such as NemoBertTokenizer
        max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
        num_samples: number of samples you want to use for the dataset.
            If -1, use all dataset. Useful for testing.
        shuffle: Shuffles the dataset after loading.
        use_cache: Enables caching to use HDFS format to store and read data from
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'label': NeuralType(('B'), LabelsType()),
        }

    def __init__(
        self,
        input_file: str,
        tokenizer: Any,
        max_seq_length: int,
        num_samples: int = -1,
        shuffle: bool = False,
        use_cache: bool = False,
    ):
        self.input_file = input_file
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.vocab_size = self.tokenizer.tokenizer.vocab_size

        if use_cache:
            data_dir, filename = os.path.split(input_file)
            vocab_size = getattr(tokenizer, "vocab_size", 0)
            tokenizer_type = type(tokenizer.tokenizer).__name__
            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}_{}".format(
                    filename[:-4], tokenizer_type, str(max_seq_length), str(vocab_size), '.hdf5'
                ),  # TODO: pylint(too-many-format-args)
            )

        if use_cache and os.path.exists(cached_features_file):
            self.load_cached_features(cached_features_file)
        else:
            with open(input_file, "r") as f:
                labels, all_sent_subtokens = [], []
                sent_lengths = []
                too_long_count = 0

                lines = f.readlines()[1:]
                logging.info(f'{input_file}: {len(lines)}')

                if shuffle or num_samples > -1:
                    random.seed(0)
                    random.shuffle(lines)
                    if num_samples > 0:
                        lines = lines[:num_samples]

                for index, line in enumerate(lines):
                    if index % 20000 == 0:
                        logging.debug(f"Processing line {index}/{len(lines)}")

                    line_splited = line.strip().split()
                    label = int(line_splited[-1])
                    labels.append(label)
                    sent_words = line_splited[:-1]
                    sent_subtokens = [tokenizer.cls_token]

                    for word in sent_words:
                        word_tokens = tokenizer.text_to_tokens(word)
                        sent_subtokens.extend(word_tokens)

                    sent_subtokens.append(tokenizer.sep_token)

                    all_sent_subtokens.append(sent_subtokens)
                    sent_lengths.append(len(sent_subtokens))
            get_stats(sent_lengths)

            for i in range(len(all_sent_subtokens)):
                if len(all_sent_subtokens[i]) > max_seq_length:
                    shorten_sent = all_sent_subtokens[i][-max_seq_length + 1 :]
                    all_sent_subtokens[i] = [tokenizer.cls_token] + shorten_sent
                    too_long_count += 1

            logging.info(
                f'{too_long_count} out of {len(sent_lengths)} \
                        sentences with more than {max_seq_length} subtokens.'
            )

            self.convert_sequences_to_features(all_sent_subtokens, labels, tokenizer, max_seq_length)

            if self.use_cache:
                self.cache_features(cached_features_file, self.features)

                # update self.features to use features from hdf5
                self.load_cached_features(cached_features_file)

    def __len__(self):
        if self.use_cache:
            return len(self.features[0])
        else:
            return len(self.features)

    def __getitem__(self, idx):
        if self.use_cache:
            return (self.features[0][idx], self.features[1][idx], self.features[2][idx], self.features[3][idx])
        else:
            feature = self.features[idx]
            return (
                np.array(feature.input_ids),
                np.array(feature.segment_ids),
                np.array(feature.input_mask, dtype=np.long),
                feature.label,
            )

    def convert_sequences_to_features(self, all_sent_subtokens, labels, tokenizer, max_seq_length):
        """Loads a data file into a list of `InputBatch`s.
        """

        self.features = []
        for sent_id in range(len(all_sent_subtokens)):
            sent_subtokens = all_sent_subtokens[sent_id]
            label = labels[sent_id]

            input_ids = [tokenizer.tokens_to_ids(t) for t in sent_subtokens]

            # The mask has 1 for real tokens and 0 for padding tokens.
            # Only real tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
            segment_ids = [0] * max_seq_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length

            if sent_id < 5:
                logging.info("*** Example ***")
                logging.info("example_index: %s" % sent_id)
                logging.info("subtokens: %s" % " ".join(sent_subtokens))
                logging.info("label: %s" % label)
                logging.info("input_ids: %s" % list2str(input_ids))
                logging.info("input_mask: %s" % list2str(input_mask))

            self.features.append(
                InputFeatures(
                    sent_id=sent_id, label=label, input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                )
            )

    def cache_features(self, cached_features_file, features):
        len_features = len(features)
        input_ids_array = np.zeros((len_features, self.max_seq_length))
        segment_ids_array = np.zeros((len_features, self.max_seq_length))
        input_mask_array = np.zeros((len_features, self.max_seq_length))
        labels_array = np.zeros((len_features,))

        for idx in range(len_features):
            input_ids_array[idx] = features[idx].input_ids
            segment_ids_array[idx] = features[idx].segment_ids
            input_mask_array[idx] = features[idx].input_mask
            labels_array[idx] = features[idx].label

        f = h5py.File(cached_features_file, mode='w')
        f.create_dataset('input_ids', data=input_ids_array)
        f.create_dataset('segment_ids', data=segment_ids_array)
        f.create_dataset('input_mask', data=input_mask_array)
        f.create_dataset('label', data=labels_array)
        f.close()

    def load_cached_features(self, cached_features_file):
        f = h5py.File(cached_features_file, 'r')
        keys = ['input_ids', 'segment_ids', 'input_mask', 'label']
        self.features = [np.asarray(f[key], dtype=np.long) for key in keys]
        f.close()
        logging.info(f'features restored from {cached_features_file}')

        if self.shuffle:
            np.random.seed(0)
            idx = np.arange(len(self))
            np.random.shuffle(idx)  # shuffle idx in place
            shuffled_features = [arr[idx] for arr in self.features]
            self.features = shuffled_features

        if self.num_samples > 0:
            truncated_features = [arr[0 : self.num_samples] for arr in self.features]
            self.features = truncated_features


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, sent_id, label, input_ids, input_mask, segment_ids):
        self.sent_ids = sent_id
        self.label = label
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
