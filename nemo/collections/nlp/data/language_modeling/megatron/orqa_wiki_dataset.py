# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""Wikipedia dataset from DPR code for ORQA."""

import csv
import random
from abc import ABC

import numpy as np
import torch
from megatron import get_args, get_tokenizer, mpu, print_rank_0
from torch.utils.data import Dataset

from nemo.collections.nlp.data.language_modeling.megatron.biencoder_dataset_utils import make_attention_mask


def get_open_retrieval_wiki_dataset():
    args = get_args()
    tokenizer = get_tokenizer()

    dataset = OpenRetrievalEvidenceDataset(
        '2018 Wikipedia from DPR codebase', 'evidence', args.evidence_data_path, tokenizer, args.retriever_seq_length
    )
    return dataset


def get_open_retrieval_batch(data_iterator):
    # Items and their type.
    keys = ['row_id', 'context', 'context_mask', 'context_types', 'context_pad_mask']
    datatype = torch.int64

    # Broadcast data.
    data = None if data_iterator is None else next(data_iterator)
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    row_id = data_b['row_id'].long()
    context = data_b['context'].long()

    # TODO: make the context mask a binary one
    context_mask = data_b['context_mask'] < 0.5

    context_types = data_b['context_types'].long()
    context_pad_mask = data_b['context_pad_mask'].long()

    return row_id, context, context_mask, context_types, context_pad_mask


def build_tokens_types_paddings_from_text(row, tokenizer, max_seq_length):
    """Build token types and paddings, trim if needed, and pad if needed."""

    title_ids = tokenizer.tokenize(row['title'])
    context_ids = tokenizer.tokenize(row['text'])

    # Appending the title of the context at front
    extended_context_ids = title_ids + [tokenizer.sep_id] + context_ids

    context_ids, context_types, context_pad_mask = build_tokens_types_paddings_from_ids(
        extended_context_ids, max_seq_length, tokenizer.cls, tokenizer.sep, tokenizer.pad
    )

    return context_ids, context_types, context_pad_mask


# noinspection DuplicatedCode
def build_tokens_types_paddings_from_ids(text_ids, max_seq_length, cls_id, sep_id, pad_id):
    """Build token types and paddings, trim if needed, and pad if needed."""
    enc_ids = []
    tokentypes_enc = []

    # [CLS].
    enc_ids.append(cls_id)
    tokentypes_enc.append(0)

    # A.
    len_src = len(text_ids)
    enc_ids.extend(text_ids)
    tokentypes_enc.extend([0] * len_src)

    # Cap the size.
    if len(enc_ids) > max_seq_length - 1:
        enc_ids = enc_ids[0 : max_seq_length - 1]
        tokentypes_enc = tokentypes_enc[0 : max_seq_length - 1]

    # [SEP].
    enc_ids.append(sep_id)
    tokentypes_enc.append(0)

    num_tokens_enc = len(enc_ids)
    # Padding.
    padding_length = max_seq_length - len(enc_ids)
    if padding_length > 0:
        enc_ids.extend([pad_id] * padding_length)
        tokentypes_enc.extend([pad_id] * padding_length)

    pad_mask = ([1] * num_tokens_enc) + ([0] * padding_length)
    pad_mask = np.array(pad_mask, dtype=np.int64)

    return enc_ids, tokentypes_enc, pad_mask


def build_sample(row_id, context_ids, context_types, context_pad_mask):
    """Convert to numpy and return a sample consumed by the batch producer."""

    context_ids = np.array(context_ids, dtype=np.int64)
    context_types = np.array(context_types, dtype=np.int64)
    context_mask = make_attention_mask(context_ids, context_ids)

    sample = {
        'row_id': row_id,
        'context': context_ids,
        'context_mask': context_mask,
        'context_types': context_types,
        'context_pad_mask': context_pad_mask,
    }
    return sample


class OpenRetrievalEvidenceDataset(ABC, Dataset):
    """Open Retrieval Evidence dataset class."""

    def __init__(self, task_name, dataset_name, datapath, tokenizer, max_seq_length):
        # Store inputs.
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        print_rank_0(' > building {} dataset for {}:'.format(self.task_name, self.dataset_name))
        # Process the files.
        print_rank_0(datapath)
        self.samples, self.id2text = self.process_samples_from_single_path(datapath)

        args = get_args()
        if args.sample_rate < 1:  # subsample
            k = int(len(self.samples) * args.sample_rate)
            self.samples = random.sample(self.samples, k)

        print_rank_0('  >> total number of samples: {}'.format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]

        context_ids, context_types, context_pad_mask = build_tokens_types_paddings_from_text(
            row, self.tokenizer, self.max_seq_length
        )

        sample = build_sample(row['doc_id'], context_ids, context_types, context_pad_mask)
        return sample

    @staticmethod
    def process_samples_from_single_path(filename):
        print_rank_0(' > Processing {} ...'.format(filename))
        total = 0

        rows = []
        id2text = {}

        with open(filename) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            next(reader, None)  # skip the headers
            for row in reader:
                # file format: doc_id, doc_text, title
                doc_id = int(row[0])
                text = row[1]
                title = row[2]

                rows.append({'doc_id': doc_id, 'text': text, 'title': title})

                assert doc_id not in id2text
                id2text[doc_id] = (text, title)

                total += 1
                if total % 100000 == 0:
                    print_rank_0('  > processed {} rows so far ...'.format(total))

        print_rank_0(' >> processed {} samples.'.format(len(rows)))
        return rows, id2text
