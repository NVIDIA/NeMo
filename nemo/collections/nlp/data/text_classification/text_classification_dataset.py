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

import os
import random
from typing import Dict, Optional, List
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

import numpy as np
import pickle
import torch

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
        tokenizer: tokenizer object such as AutoTokenizer
        max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
        num_samples: number of samples you want to use for the dataset.
            If -1, use all dataset. Useful for testing.
        shuffle: Shuffles the dataset after loading.
        use_cache: Enables caching to use pickle format to store and read data from
        pad_id: the id to be used for padding
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'label': NeuralType(('B',), LabelsType()),
        }

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        max_seq_length: int,
        input_file: str = None,
        queries: List[str] = None,
        num_samples: int = -1,
        shuffle: bool = False,
        use_cache: bool = False,
        pad_id: int = 0,
    ):

        if not input_file and not queries:
            raise ValueError("Either input_file or queries should be passed to the text classification dataset.")

        self.input_file = input_file
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_id = pad_id

        if input_file and use_cache:
            data_dir, filename = os.path.split(input_file)
            vocab_size = getattr(tokenizer, "vocab_size", 0)
            tokenizer_name = tokenizer.tokenizer_name
            cached_features_file = os.path.join(
                data_dir,
                f"cached_{filename}_{tokenizer_name}_{max_seq_length}_{vocab_size}_{num_samples}_{pad_id}_{shuffle}.pkl"
            )

        if input_file and use_cache and os.path.exists(cached_features_file):
            logging.warning(f"Reading and processing the data file {input_file} is skipped as caching is enabled and a cache file {cached_features_file} already exists.\nYou may need to delete the cache file if any of the processing parameters (eg. tokenizer) or the data are updated.")
            self.features = self.load_cached_features(cached_features_file)
        else:
            if input_file:
                if not os.path.exists(input_file):
                    raise FileNotFoundError(f'Data file {input_file} not found!')

                with open(input_file, "r") as f:
                    labels, all_sents = [], []
                    lines = f.readlines(num_samples + 1)
                    logging.info(f'Read {len(lines)} examples from {input_file}.')
                    if shuffle:
                        random.shuffle(lines)

                    for index, line in enumerate(lines):
                        if index % 20000 == 0:
                            logging.debug(f"Processing line {index}/{len(lines)}")

                        line_splited = line.strip().split()
                        label = int(line_splited[-1])
                        labels.append(label)
                        sent_words = line_splited[:-1]
                        all_sents.append(sent_words)
            else:
                all_sents = queries
            self.features = self.get_features(all_sents, tokenizer, max_seq_length, labels)

        if input_file and use_cache and not os.path.exists(cached_features_file):
            logging.warning(f"Processed data read from {input_file} is stored in {cached_features_file} as caching feature is enabled.")
            self.cache_features(cached_features_file, self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    def _collate_fn(self, batch, pad_id=0):
        """collate batch of input_ids, segment_ids, input_mask, and label
        Args:
            batch:  A list of tuples of (input_ids, segment_ids, input_mask, label).
        """
        max_length = 0
        for input_ids, segment_ids, input_mask, label in batch:
            if len(input_ids) > max_length:
                max_length = len(input_ids)

        padded_input_ids = []
        padded_segment_ids = []
        padded_input_mask = []
        labels = []
        for input_ids, segment_ids, input_mask, label in batch:
            if len(input_ids) < max_length:
                pad_width = max_length - len(input_ids)
                padded_input_ids.append(np.pad(input_ids, pad_width=[0, pad_width], constant_values=pad_id))
                padded_segment_ids.append(np.pad(segment_ids, pad_width=[0, pad_width], constant_values=pad_id))
                padded_input_mask.append(np.pad(input_mask, pad_width=[0, pad_width], constant_values=pad_id))
            else:
                padded_input_ids.append(input_ids)
                padded_segment_ids.append(segment_ids)
                padded_input_mask.append(input_mask)
            labels.append(label)

        return torch.LongTensor(padded_input_ids), torch.LongTensor(padded_segment_ids), torch.LongTensor(padded_input_mask), torch.LongTensor(labels)

    @staticmethod
    def get_features(all_sents, tokenizer, max_seq_length, labels=None):
        """Encode a list of sentences into a list of tuples of (input_ids, segment_ids, input_mask, label)."""
        features = []
        sent_lengths = []
        too_long_count = 0
        for sent_id, sent in enumerate(all_sents):
            sent_subtokens = [tokenizer.cls_token]
            for word in sent:
                word_tokens = tokenizer.text_to_tokens(word)
                sent_subtokens.extend(word_tokens)

            if len(sent_subtokens) + 1 > max_seq_length:
                sent_subtokens = sent_subtokens[:max_seq_length]
                too_long_count += 1

            sent_subtokens.append(tokenizer.sep_token)
            sent_lengths.append(len(sent_subtokens))

            input_ids = [tokenizer.tokens_to_ids(t) for t in sent_subtokens]

            # The mask has 1 for real tokens and 0 for padding tokens.
            # Only real tokens are attended to.
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)

            if sent_id < 5:
                logging.info("*** Example ***")
                logging.info(f"example {sent_id}: {sent}")
                logging.info("subtokens: %s" % " ".join(sent_subtokens))
                logging.info("input_ids: %s" % list2str(input_ids))
                logging.info("segment_ids: %s" % list2str(segment_ids))
                logging.info("input_mask: %s" % list2str(input_mask))
                logging.info("label: %s" % labels[sent_id] if labels else "**Not Provided**")

            label = labels[sent_id] if labels else -1
            features.append([np.asarray(input_ids), np.asarray(segment_ids), np.asarray(input_mask), label])

        logging.info(
            f'Found {too_long_count} out of {len(all_sents)} sentences with more than {max_seq_length} subtokens. '
            f'Truncated long sentences from the end.'
        )

        get_stats(sent_lengths)
        return features

    @staticmethod
    def cache_features(cached_features_file, features):
        with open(cached_features_file, 'wb') as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_cached_features(cached_features_file):
        with open(cached_features_file, "rb") as input_file:
            features = pickle.load(input_file)
        return features
