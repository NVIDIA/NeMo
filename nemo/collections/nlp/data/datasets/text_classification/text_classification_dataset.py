# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

"""
Utility functions for Token Classification NLP tasks
Some parts of this code were adapted from the HuggingFace library at
https://github.com/huggingface/pytorch-pretrained-BERT
"""

import os
import random

import h5py
import numpy as np
from torch.utils.data import Dataset

from nemo import logging
from nemo.collections.nlp.data.datasets.datasets_utils.data_preprocessing import get_stats
from nemo.collections.nlp.utils.callback_utils import list2str

__all__ = ['BertTextClassificationDataset']


class BertTextClassificationDataset(Dataset):
    """A dataset class that converts from raw data to
    a dataset that can be used by DataLayerNM.

    Args:
        input_file (str): file to sequence + label.
            the first line is header (sentence [tab] label)
            each line should be [sentence][tab][label]
        max_seq_length (int): max sequence length minus 2 for [CLS] and [SEP]
        tokenizer (Tokenizer): such as BertTokenizer
        num_samples (int): number of samples you want to use for the dataset.
            If -1, use all dataset. Useful for testing.
    """

    def __init__(
        self, input_file, max_seq_length, tokenizer, num_samples=-1, shuffle=False, use_cache=False,
    ):

        self.input_file = input_file
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.use_cache = use_cache
        self.shuffle = shuffle
        self.vocab_size = self.tokenizer.vocab_size

        if use_cache:
            data_dir = os.path.dirname(input_file)
            filename = os.path.basename(input_file)
            filename = filename[:-4]
            hdf5_path = os.path.join(data_dir, f'{filename}_features.hdf5')

        if use_cache and os.path.exists(hdf5_path):
            self.load_cached_features(hdf5_path)

        else:
            with open(input_file, "r") as f:
                sent_labels, all_sent_subtokens = [], []
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

                    sent_label = int(line.split()[-1])
                    sent_labels.append(sent_label)
                    sent_words = line.strip().split()[:-1]
                    sent_subtokens = ['[CLS]']

                    for word in sent_words:
                        word_tokens = tokenizer.tokenize(word)
                        sent_subtokens.extend(word_tokens)

                    sent_subtokens.append('[SEP]')

                    all_sent_subtokens.append(sent_subtokens)
                    sent_lengths.append(len(sent_subtokens))

            get_stats(sent_lengths)

            for i in range(len(all_sent_subtokens)):
                if len(all_sent_subtokens[i]) > max_seq_length:
                    shorten_sent = all_sent_subtokens[i][-max_seq_length + 1 :]
                    all_sent_subtokens[i] = ['[CLS]'] + shorten_sent
                    too_long_count += 1

            logging.info(
                f'{too_long_count} out of {len(sent_lengths)} \
                        sentences with more than {max_seq_length} subtokens.'
            )

            self.convert_sequences_to_features(all_sent_subtokens, sent_labels, tokenizer, max_seq_length)

            if self.use_cache:
                self.cache_features(hdf5_path, self.features)

                # update self.features to use features from hdf5
                self.load_cached_features(hdf5_path)

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
                feature.sent_label,
            )

    def convert_sequences_to_features(self, all_sent_subtokens, sent_labels, tokenizer, max_seq_length):
        """Loads a data file into a list of `InputBatch`s.
        """

        self.features = []
        for sent_id in range(len(all_sent_subtokens)):
            sent_subtokens = all_sent_subtokens[sent_id]
            sent_label = sent_labels[sent_id]

            input_ids = [tokenizer._convert_token_to_id(t) for t in sent_subtokens]

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

            if sent_id == 0:
                logging.info("*** Example ***")
                logging.info("example_index: %s" % sent_id)
                logging.info("subtokens: %s" % " ".join(sent_subtokens))
                logging.info("sent_label: %s" % sent_label)
                logging.info("input_ids: %s" % list2str(input_ids))
                logging.info("input_mask: %s" % list2str(input_mask))

            self.features.append(
                InputFeatures(
                    sent_id=sent_id,
                    sent_label=sent_label,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                )
            )

    def cache_features(self, hdf5_path, features):
        len_features = len(features)
        input_ids_array = np.zeros((len_features, self.max_seq_length))
        segment_ids_array = np.zeros((len_features, self.max_seq_length))
        input_mask_array = np.zeros((len_features, self.max_seq_length))
        sent_labels_array = np.zeros((len_features,))

        for idx in range(len_features):
            input_ids_array[idx] = features[idx].input_ids
            segment_ids_array[idx] = features[idx].segment_ids
            input_mask_array[idx] = features[idx].input_mask
            sent_labels_array[idx] = features[idx].sent_label

        f = h5py.File(hdf5_path, mode='w')
        f.create_dataset('input_ids', data=input_ids_array)
        f.create_dataset('segment_ids', data=segment_ids_array)
        f.create_dataset('input_mask', data=input_mask_array)
        f.create_dataset('sent_labels', data=sent_labels_array)
        f.close()

    def load_cached_features(self, hdf5_path):
        f = h5py.File(hdf5_path, 'r')
        keys = ['input_ids', 'segment_ids', 'input_mask', 'sent_labels']
        self.features = [np.asarray(f[key], dtype=np.long) for key in keys]
        f.close()
        logging.info(f'features restored from {hdf5_path}')

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

    def __init__(self, sent_id, sent_label, input_ids, input_mask, segment_ids):
        self.sent_id = sent_id
        self.sent_label = sent_label
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
