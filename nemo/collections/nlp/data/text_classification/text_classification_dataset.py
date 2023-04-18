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
import pickle
import random
from typing import Dict, List, Optional

import numpy as np
import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.data_utils.data_preprocessing import (
    fill_class_weights,
    get_freq_weights,
    get_label_stats,
    get_stats,
)
from nemo.collections.nlp.parts.utils_funcs import list2str
from nemo.core.classes import Dataset
from nemo.core.neural_types import ChannelType, LabelsType, MaskType, NeuralType
from nemo.utils import logging
from nemo.utils.env_var_parsing import get_envint

__all__ = ['TextClassificationDataset', 'calc_class_weights']


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
        input_file: str = None,
        queries: List[str] = None,
        max_seq_length: int = -1,
        num_samples: int = -1,
        shuffle: bool = False,
        use_cache: bool = False,
    ):
        if not input_file and not queries:
            raise ValueError("Either input_file or queries should be passed to the text classification dataset.")

        if input_file and not os.path.exists(input_file):
            raise FileNotFoundError(
                f'Data file `{input_file}` not found! Each line of the data file should contain text sequences, where '
                f'words are separated with spaces and the label separated by [TAB] following this format: '
                f'[WORD][SPACE][WORD][SPACE][WORD][TAB][LABEL]'
            )

        self.input_file = input_file
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_id = tokenizer.pad_id

        self.features = None
        labels, all_sents = [], []
        if input_file:
            data_dir, filename = os.path.split(input_file)
            vocab_size = getattr(tokenizer, "vocab_size", 0)
            tokenizer_name = tokenizer.name
            cached_features_file = os.path.join(
                data_dir,
                f"cached_{filename}_{tokenizer_name}_{max_seq_length}_{vocab_size}_{num_samples}_{self.pad_id}_{shuffle}.pkl",
            )

            if get_envint("LOCAL_RANK", 0) == 0:
                if use_cache and os.path.exists(cached_features_file):
                    logging.warning(
                        f"Processing of {input_file} is skipped as caching is enabled and a cache file "
                        f"{cached_features_file} already exists."
                    )
                    logging.warning(
                        f"You may need to delete the cache file if any of the processing parameters (eg. tokenizer) or "
                        f"the data are updated."
                    )
                else:
                    with open(input_file, "r") as f:
                        lines = f.readlines()
                        logging.info(f'Read {len(lines)} examples from {input_file}.')
                        if num_samples > 0:
                            lines = lines[:num_samples]
                            logging.warning(
                                f"Parameter 'num_samples' is set, so just the first {len(lines)} examples are kept."
                            )

                        if shuffle:
                            random.shuffle(lines)

                        for index, line in enumerate(lines):
                            if index % 20000 == 0:
                                logging.debug(f"Processing line {index}/{len(lines)}")
                            line_splited = line.strip().split()
                            try:
                                label = int(line_splited[-1])
                            except ValueError:
                                logging.debug(f"Skipping line {line}")
                                continue
                            labels.append(label)
                            sent_words = line_splited[:-1]
                            all_sents.append(sent_words)
                    verbose = True

                    self.features = self.get_features(
                        all_sents=all_sents,
                        tokenizer=tokenizer,
                        max_seq_length=max_seq_length,
                        labels=labels,
                        verbose=verbose,
                    )
                    with open(cached_features_file, 'wb') as out_file:
                        pickle.dump(self.features, out_file, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            for query in queries:
                all_sents.append(query.strip().split())
            labels = [-1] * len(all_sents)
            verbose = False
            self.features = self.get_features(
                all_sents=all_sents, tokenizer=tokenizer, max_seq_length=max_seq_length, labels=labels, verbose=verbose
            )

        # wait until the master process writes to the processed data files
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if input_file:
            with open(cached_features_file, "rb") as input_file:
                self.features = pickle.load(input_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    def _collate_fn(self, batch):
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
                padded_input_ids.append(np.pad(input_ids, pad_width=[0, pad_width], constant_values=self.pad_id))
                padded_segment_ids.append(np.pad(segment_ids, pad_width=[0, pad_width], constant_values=self.pad_id))
                padded_input_mask.append(np.pad(input_mask, pad_width=[0, pad_width], constant_values=self.pad_id))
            else:
                padded_input_ids.append(input_ids)
                padded_segment_ids.append(segment_ids)
                padded_input_mask.append(input_mask)
            labels.append(label)

        return (
            torch.LongTensor(padded_input_ids),
            torch.LongTensor(padded_segment_ids),
            torch.LongTensor(padded_input_mask),
            torch.LongTensor(labels),
        )

    @staticmethod
    def get_features(all_sents, tokenizer, max_seq_length, labels=None, verbose=True):
        """Encode a list of sentences into a list of tuples of (input_ids, segment_ids, input_mask, label)."""
        features = []
        sent_lengths = []
        too_long_count = 0
        for sent_id, sent in enumerate(all_sents):
            if sent_id % 1000 == 0:
                logging.debug(f"Encoding sentence {sent_id}/{len(all_sents)}")
            sent_subtokens = [tokenizer.cls_token]
            for word in sent:
                word_tokens = tokenizer.text_to_tokens(word)
                sent_subtokens.extend(word_tokens)

            if max_seq_length > 0 and len(sent_subtokens) + 1 > max_seq_length:
                sent_subtokens = sent_subtokens[: max_seq_length - 1]
                too_long_count += 1

            sent_subtokens.append(tokenizer.sep_token)
            sent_lengths.append(len(sent_subtokens))

            input_ids = [tokenizer.tokens_to_ids(t) for t in sent_subtokens]

            # The mask has 1 for real tokens and 0 for padding tokens.
            # Only real tokens are attended to.
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)

            if verbose and sent_id < 2:
                logging.info("*** Example ***")
                logging.info(f"example {sent_id}: {sent}")
                logging.info("subtokens: %s" % " ".join(sent_subtokens))
                logging.info("input_ids: %s" % list2str(input_ids))
                logging.info("segment_ids: %s" % list2str(segment_ids))
                logging.info("input_mask: %s" % list2str(input_mask))
                logging.info("label: %s" % labels[sent_id] if labels else "**Not Provided**")

            label = labels[sent_id] if labels else -1
            features.append([np.asarray(input_ids), np.asarray(segment_ids), np.asarray(input_mask), label])

        if max_seq_length > -1 and too_long_count > 0:
            logging.warning(
                f'Found {too_long_count} out of {len(all_sents)} sentences with more than {max_seq_length} subtokens. '
                f'Truncated long sentences from the end.'
            )
        if verbose:
            get_stats(sent_lengths)
        return features


def calc_class_weights(file_path: str, num_classes: int):
    """
    iterates over a data file and calculate the weights of each class to be used for class_balancing
    Args:
        file_path: path to the data file
        num_classes: number of classes in the dataset
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find data file {file_path} to calculate the class weights!")

    with open(file_path, 'r') as f:
        input_lines = f.readlines()

    labels = []
    for input_line in input_lines:
        parts = input_line.strip().split()
        try:
            label = int(parts[-1])
        except ValueError:
            raise ValueError(
                f'No numerical labels found for {file_path}. Labels should be integers and separated by [TAB] at the end of each line.'
            )
        labels.append(label)

    logging.info(f'Calculating stats of {file_path}...')
    total_sents, sent_label_freq, max_id = get_label_stats(labels, f'{file_path}_sentence_stats.tsv', verbose=False)
    if max_id >= num_classes:
        raise ValueError(f'Found an invalid label in {file_path}! Labels should be from [0, num_classes-1].')

    class_weights_dict = get_freq_weights(sent_label_freq)

    logging.info(f'Total Sentence: {total_sents}')
    logging.info(f'Sentence class frequencies: {sent_label_freq}')

    logging.info(f'Class Weights: {class_weights_dict}')
    class_weights = fill_class_weights(weights=class_weights_dict, max_id=num_classes - 1)

    return class_weights
