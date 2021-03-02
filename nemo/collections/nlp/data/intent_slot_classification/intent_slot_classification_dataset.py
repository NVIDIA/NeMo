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

import math
from typing import Any, Dict, Optional
import random

import numpy as np
from torch.utils.data import Sampler

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.data_utils import get_stats
from nemo.core.classes import Dataset
from nemo.core.neural_types import ChannelType, LabelsType, MaskType, NeuralType
from nemo.utils import logging

__all__ = ['IntentSlotClassificationDataset', 'IntentSlotInferenceDataset']


def get_features(
    queries,
    max_seq_length,
    tokenizer,
    pad_label=128,
    raw_slots=None,
    ignore_extra_tokens=False,
    ignore_start_end=False,
):
    all_subtokens = []
    all_loss_mask = []
    all_subtokens_mask = []
    all_segment_ids = []
    all_input_ids = []
    all_input_mask = []
    sent_lengths = []
    all_slots = []

    with_label = False
    if raw_slots is not None:
        with_label = True

    for i, query in enumerate(queries):
        words = query.strip().split()
        subtokens = [tokenizer.cls_token]
        loss_mask = [1 - ignore_start_end]
        subtokens_mask = [0]
        if with_label:
            slots = [pad_label]

        for j, word in enumerate(words):
            word_tokens = tokenizer.text_to_tokens(word)
            subtokens.extend(word_tokens)

            loss_mask.append(1)
            loss_mask.extend([int(not ignore_extra_tokens)] * (len(word_tokens) - 1))

            subtokens_mask.append(1)
            subtokens_mask.extend([0] * (len(word_tokens) - 1))

            if with_label:
                slots.extend([raw_slots[i][j]] * len(word_tokens))

        subtokens.append(tokenizer.sep_token)
        loss_mask.append(1 - ignore_start_end)
        subtokens_mask.append(0)
        sent_lengths.append(len(subtokens))
        all_subtokens.append(subtokens)
        all_loss_mask.append(loss_mask)
        all_subtokens_mask.append(subtokens_mask)
        all_input_mask.append([1] * len(subtokens))
        if with_label:
            slots.append(pad_label)
            all_slots.append(slots)

    max_seq_length_data = max(sent_lengths)
    max_seq_length = min(max_seq_length, max_seq_length_data) if max_seq_length > 0 else max_seq_length_data
    logging.info(f'Setting max length to: {max_seq_length}')
    get_stats(sent_lengths)
    too_long_count = 0

    for i, subtokens in enumerate(all_subtokens):
        if len(subtokens) > max_seq_length:
            subtokens = [tokenizer.cls_token] + subtokens[-max_seq_length + 1 :]
            all_input_mask[i] = [1] + all_input_mask[i][-max_seq_length + 1 :]
            all_loss_mask[i] = [1 - ignore_start_end] + all_loss_mask[i][-max_seq_length + 1 :]
            all_subtokens_mask[i] = [0] + all_subtokens_mask[i][-max_seq_length + 1 :]

            if with_label:
                all_slots[i] = [pad_label] + all_slots[i][-max_seq_length + 1 :]
            too_long_count += 1

        all_input_ids.append([tokenizer.tokens_to_ids(t) for t in subtokens])

        if len(subtokens) < max_seq_length:
            extra = max_seq_length - len(subtokens)
            all_input_ids[i] = all_input_ids[i] + [0] * extra
            all_loss_mask[i] = all_loss_mask[i] + [0] * extra
            all_subtokens_mask[i] = all_subtokens_mask[i] + [0] * extra
            all_input_mask[i] = all_input_mask[i] + [0] * extra

            if with_label:
                all_slots[i] = all_slots[i] + [pad_label] * extra

        all_segment_ids.append([0] * max_seq_length)

    logging.info(f'{too_long_count} are longer than {max_seq_length}')

    # May be useful for debugging
    logging.debug("*** Some Examples of Processed Data ***")
    for i in range(min(len(all_input_ids), 5)):
        logging.debug("i: %s" % (i))
        logging.debug("subtokens: %s" % " ".join(list(map(str, all_subtokens[i]))))
        logging.debug("loss_mask: %s" % " ".join(list(map(str, all_loss_mask[i]))))
        logging.debug("input_mask: %s" % " ".join(list(map(str, all_input_mask[i]))))
        logging.debug("subtokens_mask: %s" % " ".join(list(map(str, all_subtokens_mask[i]))))
        if with_label:
            logging.debug("slots_label: %s" % " ".join(list(map(str, all_slots[i]))))

    return (all_input_ids, all_segment_ids, all_input_mask, all_loss_mask, all_subtokens_mask, all_slots)


class IntentSlotClassificationDataset(Dataset):
    """
    Creates dataset to use for the task of joint intent
    and slot classification with pretrained model.

    Converts from raw data to an instance that can be used by
    NMDataLayer.

    For dataset to use during inference without labels, see
    IntentSlotDataset.

    Args:
        input_file: file to sequence + label. the first line is header (sentence [tab] label)
            each line should be [sentence][tab][label]
        slot_file: file to slot labels, each line corresponding to slot labels for a sentence in input_file. No header.
        max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
        tokenizer: such as NemoBertTokenizer
        num_samples: number of samples you want to use for the dataset. If -1, use all dataset. Useful for testing.
        pad_label: pad value use for slot labels. by default, it's the neutral label.
        ignore_extra_tokens: whether to ignore extra tokens in the loss_mask.
        ignore_start_end: whether to ignore bos and eos tokens in the loss_mask.
        do_lower_case: convert query to lower case or not
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'loss_mask': NeuralType(('B', 'T'), MaskType()),
            'subtokens_mask': NeuralType(('B', 'T'), MaskType()),
            'intent_labels': NeuralType(('B'), LabelsType()),
            'slot_labels': NeuralType(('B', 'T'), LabelsType()),
        }

    def __init__(
        self,
        input_file: str,
        slot_file: str,
        max_seq_length: int,
        tokenizer: TokenizerSpec,
        num_samples: int = -1,
        pad_label: int = 128,
        ignore_extra_tokens: bool = False,
        ignore_start_end: bool = False,
        do_lower_case: bool = False,
    ):
        if num_samples == 0:
            raise ValueError("num_samples has to be positive", num_samples)

        with open(slot_file, 'r') as f:
            slot_lines = f.readlines()

        with open(input_file, 'r') as f:
            input_lines = f.readlines()[1:]

        assert len(slot_lines) == len(input_lines)

        dataset = list(zip(slot_lines, input_lines))

        if num_samples > 0:
            dataset = dataset[:num_samples]

        raw_slots, queries, raw_intents = [], [], []
        for slot_line, input_line in dataset:
            raw_slots.append([int(slot) for slot in slot_line.strip().split()])
            parts = input_line.strip().split()
            raw_intents.append(int(parts[-1]))
            query = ' '.join(parts[:-1])
            if do_lower_case:
                query = query.lower()
            queries.append(query)

        features = get_features(
            queries,
            max_seq_length,
            tokenizer,
            pad_label=pad_label,
            raw_slots=raw_slots,
            ignore_extra_tokens=ignore_extra_tokens,
            ignore_start_end=ignore_start_end,
        )
        self.all_input_ids = features[0]
        self.all_segment_ids = features[1]
        self.all_input_mask = features[2]
        self.all_loss_mask = features[3]
        self.all_subtokens_mask = features[4]
        self.all_slots = features[5]
        self.all_intents = raw_intents

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        return (
            np.array(self.all_input_ids[idx]),
            np.array(self.all_segment_ids[idx]),
            np.array(self.all_input_mask[idx], dtype=np.long),
            np.array(self.all_loss_mask[idx]),
            np.array(self.all_subtokens_mask[idx]),
            self.all_intents[idx],
            np.array(self.all_slots[idx]),
        )


class IntentSlotInferenceDataset(Dataset):
    """
    Creates dataset to use for the task of joint intent
    and slot classification with pretrained model.
    This is to be used during inference only.
    It uses list of queries as the input.

    Args:
        queries (list): list of queries to run inference on
        max_seq_length (int): max sequence length minus 2 for [CLS] and [SEP]
        tokenizer (Tokenizer): such as NemoBertTokenizer
        pad_label (int): pad value use for slot labels.
            by default, it's the neutral label.

    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """
            Returns definitions of module output ports.
        """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'loss_mask': NeuralType(('B', 'T'), MaskType()),
            'subtokens_mask': NeuralType(('B', 'T'), MaskType()),
        }

    def __init__(self, queries, max_seq_length, tokenizer, do_lower_case):
        if do_lower_case:
            for idx, query in enumerate(queries):
                queries[idx] = queries[idx].lower()

        features = get_features(queries, max_seq_length, tokenizer)

        self.all_input_ids = features[0]
        self.all_segment_ids = features[1]
        self.all_input_mask = features[2]
        self.all_loss_mask = features[3]
        self.all_subtokens_mask = features[4]

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        return (
            np.array(self.all_input_ids[idx]),
            np.array(self.all_segment_ids[idx]),
            np.array(self.all_input_mask[idx], dtype=np.long),
            np.array(self.all_loss_mask[idx]),
            np.array(self.all_subtokens_mask[idx]),
        )

class IntentOnlyDataset(IntentSlotClassificationDataset):
    """
    Args:
        input_file: file with utterances and intent labels. the first line is header (sentence [tab] label)
            each line should be [sentence][tab][label]
        max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
        tokenizer: such as NemoBertTokenizer
        num_samples: number of samples you want to use for the dataset. If -1, use all dataset. Useful for testing.
        pad_label: pad value use for slot labels. by default, it's the neutral label.
        ignore_extra_tokens: whether to ignore extra tokens in the loss_mask.
        ignore_start_end: whether to ignore bos and eos tokens in the loss_mask.
        do_lower_case: convert query to lower case or not
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'loss_mask': NeuralType(('B', 'T'), MaskType()),
            'subtokens_mask': NeuralType(('B', 'T'), MaskType()),
            'intent_labels': NeuralType(('B'), LabelsType()),
            'slot_labels': NeuralType(('B', 'T'), LabelsType()),
            'data_type_indicator': NeuralType(('B'), LabelsType())
        }

    def __init__(
            self,
            input_file: str,
            max_seq_length: int,
            tokenizer: TokenizerSpec,
            num_samples: int = -1,
            pad_label: int = 128,
            ignore_extra_tokens: bool = False,
            ignore_start_end: bool = False,
            do_lower_case: bool = False,
    ):
        if num_samples == 0:
            raise ValueError("num_samples has to be positive", num_samples)

        with open(input_file, 'r') as f:
            dataset = f.readlines()[1:]

        if num_samples > 0:
            dataset = dataset[:num_samples]

        raw_slots, queries, raw_intents = [], [], []
        for input_line in dataset:
            parts = input_line.strip().split()
            raw_intents.append(int(parts[-1]))
            raw_slots.append([pad_label] * len(parts[:-1]))
            query = ' '.join(parts[:-1])
            if do_lower_case:
                query = query.lower()
            queries.append(query)

        features = get_features(
            queries,
            max_seq_length,
            tokenizer,
            pad_label=pad_label,
            raw_slots=raw_slots,
            ignore_extra_tokens=ignore_extra_tokens,
            ignore_start_end=ignore_start_end,
        )
        self.all_input_ids = features[0]
        self.all_segment_ids = features[1]
        self.all_input_mask = features[2]
        self.all_loss_mask = features[3]
        self.all_subtokens_mask = features[4]
        self.all_slots = features[5]
        self.all_intents = raw_intents

    def __getitem__(self, idx):
        return (
            np.array(self.all_input_ids[idx]),
            np.array(self.all_segment_ids[idx]),
            np.array(self.all_input_mask[idx], dtype=np.long),
            np.array(self.all_loss_mask[idx]),
            np.array(self.all_subtokens_mask[idx]),
            self.all_intents[idx],
            np.array(self.all_slots[idx]),
            1  # indicates intent_only data
        )


class SlotOnlyDataset(IntentSlotClassificationDataset):
    """
    Args:
        input_file: file with utterances only (no intent labels). No header.
        slot_file: file with slot labels, each line corresponding to slot labels for a sentence in input_file. No header.
        max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
        tokenizer: such as NemoBertTokenizer
        num_samples: number of samples you want to use for the dataset. If -1, use all dataset. Useful for testing.
        pad_label: pad value use for slot labels. by default, it's the neutral label.
        ignore_extra_tokens: whether to ignore extra tokens in the loss_mask.
        ignore_start_end: whether to ignore bos and eos tokens in the loss_mask.
        do_lower_case: convert query to lower case or not
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'loss_mask': NeuralType(('B', 'T'), MaskType()),
            'subtokens_mask': NeuralType(('B', 'T'), MaskType()),
            'intent_labels': NeuralType(('B'), LabelsType()),
            'slot_labels': NeuralType(('B', 'T'), LabelsType()),
            'data_type_indicator': NeuralType(('B'), LabelsType())
        }

    def __init__(
            self,
            input_file: str,
            slot_file: str,
            max_seq_length: int,
            tokenizer: TokenizerSpec,
            num_samples: int = -1,
            pad_label: int = 128,
            ignore_extra_tokens: bool = False,
            ignore_start_end: bool = False,
            do_lower_case: bool = False,
    ):
        if num_samples == 0:
            raise ValueError("num_samples has to be positive", num_samples)

        with open(slot_file, 'r') as f:
            slot_lines = f.readlines()

        with open(input_file, 'r') as f:
            input_lines = f.readlines()

        assert len(slot_lines) == len(input_lines)

        dataset = list(zip(slot_lines, input_lines))

        if num_samples > 0:
            dataset = dataset[:num_samples]

        raw_slots, queries, raw_intents = [], [], []
        for slot_line, input_line in dataset:
            raw_slots.append([int(slot) for slot in slot_line.strip().split()])
            parts = input_line.strip().split()
            raw_intents.append(0)  # arbitrarily using intent 0
            query = ' '.join(parts[:-1])
            if do_lower_case:
                query = query.lower()
            queries.append(query)

        features = get_features(
            queries,
            max_seq_length,
            tokenizer,
            pad_label=pad_label,
            raw_slots=raw_slots,
            ignore_extra_tokens=ignore_extra_tokens,
            ignore_start_end=ignore_start_end,
        )
        self.all_input_ids = features[0]
        self.all_segment_ids = features[1]
        self.all_input_mask = features[2]
        self.all_loss_mask = features[3]
        self.all_subtokens_mask = features[4]
        self.all_slots = features[5]
        self.all_intents = raw_intents

    def __getitem__(self, idx):
        return (
            np.array(self.all_input_ids[idx]),
            np.array(self.all_segment_ids[idx]),
            np.array(self.all_input_mask[idx], dtype=np.long),
            np.array(self.all_loss_mask[idx]),
            np.array(self.all_subtokens_mask[idx]),
            self.all_intents[idx],
            np.array(self.all_slots[idx]),
            0  # indicates slot-only data
        )

def batch(iterable, batch_size):
    """
    Yield batches from iterable.
    """
    length = len(iterable)
    for i in range(0, length, batch_size):
        yield iterable[i:min(i + batch_size, length)]


class MultiDatasetSampler(Sampler):
    """
    Given a list of datasets, yield batches; each batch only draws examples from one dataset.
    Order of examples is random within each dataset and order of which dataset each batch is from is random.
    TODO: this currently doesn't balance data between intents and slots, but it should. If one of the
    datasets is much larger than the other(s), the model will effectively place more importance on intents or slots.
    """
    def __init__(self, datasets, batch_size):
        self.datasets = datasets
        self.batch_size = batch_size

        # calculate total number of batches
        self.total_batches = 0
        for d in datasets:
            batch_count = math.ceil(len(d)/self.batch_size)
            self.total_batches += batch_count

    def __iter__(self):
        self.iterators = {}  # dataset index as key, batch iterator as value
        self.batch_sources = []  # list of dataset indices for tracking which iterator to draw from

        start = 0
        for i, dataset in enumerate(self.datasets):
            end = start + len(dataset)
            dataset_indices = list(range(start, end))
            random.shuffle(dataset_indices)
            self.iterators[i] = batch(dataset_indices, self.batch_size)
            start = end
            batch_count = math.ceil(len(dataset)/self.batch_size)
            current_batches = [i] * batch_count
            self.batch_sources += current_batches
        random.shuffle(self.batch_sources)

        for b in self.batch_sources:
            yield (next(self.iterators[b]))

    def __len__(self):
        return self.total_batches
