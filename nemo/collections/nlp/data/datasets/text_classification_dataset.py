# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
Utility functions for Token Classification NLP tasks
Some parts of this code were adapted from the HuggingFace library at
https://github.com/huggingface/pytorch-pretrained-BERT
"""

import random

import numpy as np
from torch.utils.data import Dataset

import nemo
from nemo import logging
import nemo.collections.nlp.data.datasets.joint_intent_slot_dataset
from nemo.collections.nlp.data.datasets.datasets_utils import (
    get_intent_labels,
    get_label_stats,
    get_stats,
    process_imdb,
    process_jarvis_datasets,
    process_nlu,
    process_sst_2,
    process_thucnews,
)
from nemo.collections.nlp.utils.common_nlp_utils import calc_class_weights, if_exist

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
        shuffle (bool): whether to shuffle your data.
    """

    def __init__(self, input_file, max_seq_length, tokenizer, num_samples=-1, shuffle=True):
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
        self.max_seq_length = min(max_seq_length, max(sent_lengths))

        for i in range(len(all_sent_subtokens)):
            if len(all_sent_subtokens[i]) > self.max_seq_length:
                shorten_sent = all_sent_subtokens[i][-self.max_seq_length + 1 :]
                all_sent_subtokens[i] = ['[CLS]'] + shorten_sent
                too_long_count += 1

        logging.info(
            f'{too_long_count} out of {len(sent_lengths)} \
                       sentencess with more than {max_seq_length} subtokens.'
        )

        self.convert_sequences_to_features(all_sent_subtokens, sent_labels, tokenizer, self.max_seq_length)

        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

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
                logging.info("input_ids: %s" % nemo.collections.nlp.utils.callback_utils.list2str(input_ids))
                logging.info("input_mask: %s" % nemo.collections.nlp.utils.callback_utils.list2str(input_mask))

            self.features.append(
                InputFeatures(
                    sent_id=sent_id,
                    sent_label=sent_label,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                )
            )


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, sent_id, sent_label, input_ids, input_mask, segment_ids):
        self.sent_id = sent_id
        self.sent_label = sent_label
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class SentenceClassificationDataDesc:
    def __init__(self, dataset_name, data_dir, do_lower_case):
        if dataset_name == 'sst-2':
            self.data_dir = process_sst_2(data_dir)
            self.num_labels = 2
            self.eval_file = self.data_dir + '/dev.tsv'
        elif dataset_name == 'imdb':
            self.num_labels = 2
            self.data_dir = process_imdb(data_dir, do_lower_case)
            self.eval_file = self.data_dir + '/test.tsv'
        elif dataset_name == 'thucnews':
            self.num_labels = 14
            self.data_dir = process_thucnews(data_dir)
            self.eval_file = self.data_dir + '/test.tsv'
        elif dataset_name.startswith('nlu-'):
            if dataset_name.endswith('chat'):
                self.data_dir = f'{data_dir}/ChatbotCorpus.json'
                self.num_labels = 2
            elif dataset_name.endswith('ubuntu'):
                self.data_dir = f'{data_dir}/AskUbuntuCorpus.json'
                self.num_labels = 5
            elif dataset_name.endswith('web'):
                data_dir = f'{data_dir}/WebApplicationsCorpus.json'
                self.num_labels = 8
            self.data_dir = process_nlu(data_dir, do_lower_case, dataset_name=dataset_name)
            self.eval_file = self.data_dir + '/test.tsv'
        elif dataset_name.startswith('jarvis'):
            self.data_dir = process_jarvis_datasets(
                data_dir, do_lower_case, dataset_name, modes=['train', 'test', 'eval'], ignore_prev_intent=False
            )

            intents = get_intent_labels(f'{self.data_dir}/dict.intents.csv')
            self.num_labels = len(intents)
        else:
            raise ValueError(
                "Looks like you passed a dataset name that isn't "
                "already supported by NeMo. Please make sure "
                "that you build the preprocessing method for it."
            )

        self.train_file = self.data_dir + '/train.tsv'

        for mode in ['train', 'test', 'eval']:

            if not if_exist(self.data_dir, [f'{mode}.tsv']):
                logging.info(f' Stats calculation for {mode} mode' f' is skipped as {mode}.tsv was not found.')
                continue

            input_file = f'{self.data_dir}/{mode}.tsv'
            with open(input_file, 'r') as f:
                input_lines = f.readlines()[1:]  # Skipping headers at index 0

            queries, raw_sentences = [], []
            for input_line in input_lines:
                parts = input_line.strip().split()
                raw_sentences.append(int(parts[-1]))
                queries.append(' '.join(parts[:-1]))

            infold = input_file[: input_file.rfind('/')]

            logging.info(f'Three most popular classes during {mode}ing')
            total_sents, sent_label_freq = get_label_stats(raw_sentences, infold + f'/{mode}_sentence_stats.tsv')

            if mode == 'train':
                self.class_weights = calc_class_weights(sent_label_freq)
                logging.info(f'Class weights are - {self.class_weights}')

            logging.info(f'Total Sentences - {total_sents}')
            logging.info(f'Sentence class frequencies - {sent_label_freq}')
