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

"""Pytorch Dataset for training Neural Machine Translation."""

from collections import OrderedDict

import numpy as np
from torch.utils.data import Dataset

from nemo.collections.nlp.data.datasets.datasets_utils.data_preprocessing import dataset_to_ids

__all__ = ['TranslationDataset']


class TranslationDataset(Dataset):
    def __init__(self, tokenizer_src, tokenizer_tgt, dataset_src, dataset_tgt, tokens_in_batch=1024, clean=False):

        self.src_tokenizer = tokenizer_src
        self.tgt_tokenizer = tokenizer_tgt
        self.tokens_in_batch = tokens_in_batch

        src_ids = dataset_to_ids(dataset_src, tokenizer_src)
        tgt_ids = dataset_to_ids(dataset_tgt, tokenizer_tgt)
        if clean:
            src_ids, tgt_ids = self.clean_src_and_target(src_ids, tgt_ids)
        self.batch_indices = self.pack_data_into_batches(src_ids, tgt_ids)
        self.batches = self.pad_batches(src_ids, tgt_ids, self.batch_indices)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        src_ids = self.batches[idx]["src"]
        tgt = self.batches[idx]["tgt"]
        labels = tgt[:, 1:]
        tgt_ids = tgt[:, :-1]
        src_mask = (src_ids != self.src_tokenizer.pad_id).astype(np.int32)
        tgt_mask = (tgt_ids != self.tgt_tokenizer.pad_id).astype(np.int32)
        sent_ids = self.batch_indices[idx]
        return src_ids, src_mask, tgt_ids, tgt_mask, labels, sent_ids

    def pad_batches(self, src_ids, tgt_ids, batch_indices):
        """
        Augments source and target ids in the batches with padding symbol
        to make the lengths of all sentences in the batches equal.
        """

        batches = {}
        for batch_idx, b in enumerate(batch_indices):
            src_len = max([len(src_ids[i]) for i in b])
            tgt_len = max([len(tgt_ids[i]) for i in b])
            src_ids_ = self.src_tokenizer.pad_id * np.ones((len(b), src_len), dtype=np.int)
            tgt_ids_ = self.tgt_tokenizer.pad_id * np.ones((len(b), tgt_len), dtype=np.int)
            for i, sentence_idx in enumerate(b):
                src_ids_[i][: len(src_ids[sentence_idx])] = src_ids[sentence_idx]
                tgt_ids_[i][: len(tgt_ids[sentence_idx])] = tgt_ids[sentence_idx]
            batches[batch_idx] = {"src": src_ids_, "tgt": tgt_ids_}
        return batches

    def pack_data_into_batches(self, src_ids, tgt_ids):
        """
        Takes two lists of source and target sentences, sorts them, and packs
        into batches to minimize the use of padding tokens. Returns a list of
        batches where each batch contains indices of sentences included into it
        """

        # create buckets sorted by the number of src tokens
        # each bucket is also sorted by the number of tgt tokens
        buckets = {}
        for i, src_id in enumerate(src_ids):
            src_len, tgt_len = len(src_id), len(tgt_ids[i])
            if src_len not in buckets.keys():
                buckets[src_len] = [(tgt_len, i)]
            else:
                buckets[src_len].append((tgt_len, i))

        for b_idx in buckets:
            buckets[b_idx] = sorted(buckets[b_idx])

        buckets = OrderedDict(sorted(buckets.items()))
        indices = list(buckets.keys())

        batches = [[]]
        num_batches = 0
        batch_size = 0
        i = 0
        src_len = 0
        tgt_len = 0

        while i < len(buckets.keys()):

            while buckets[indices[i]]:

                i_src = max(src_len, indices[i])
                i_tgt = max(tgt_len, buckets[indices[i]][0][0])

                try:
                    ip1_src = max(src_len, indices[i + 1])
                    ip1_tgt = max(tgt_len, buckets[indices[i + 1]][0][0])
                except IndexError:
                    ip1_src = i_src + 1
                    ip1_tgt = i_tgt + 1

                if i_src + i_tgt <= ip1_src + ip1_tgt:
                    src_len = i_src
                    tgt_len = i_tgt
                    _, idx = buckets[indices[i]].pop(0)
                else:
                    src_len = ip1_src
                    tgt_len = ip1_tgt
                    _, idx = buckets[indices[i + 1]].pop(0)

                batches[num_batches].append(idx)
                batch_size += 1

                if batch_size * (src_len + tgt_len) > self.tokens_in_batch:

                    num_examples_to_split = len(batches[num_batches])
                    batches_to_evict = 8 * ((num_examples_to_split - 1) // 8)

                    if batches_to_evict == 0:
                        batches_to_evict = num_examples_to_split

                    batches.append(batches[num_batches][batches_to_evict:])
                    batches[num_batches] = batches[num_batches][:batches_to_evict]
                    batch_size = num_examples_to_split - batches_to_evict

                    num_batches += 1
                    if batch_size > 0:
                        src_len = max([len(src_ids[j]) for j in batches[num_batches]])
                        tgt_len = max([len(tgt_ids[j]) for j in batches[num_batches]])
                    else:
                        src_len = 0
                        tgt_len = 0
                    break

            if not buckets[indices[i]]:
                i = i + 1

        if not batches[-1]:
            batches.pop(-1)

        return batches

    def clean_src_and_target(
        self, src_ids, tgt_ids, max_tokens=128, min_tokens=3, max_tokens_diff=25, max_tokens_ratio=2.5
    ):
        """
        Cleans source and target sentences to get rid of noisy data.
        Specifically, a pair of sentences is removed if
          -- either source or target is longer than *max_tokens*
          -- either source or target is shorter than *min_tokens*
          -- absolute difference between source and target is larger than
             *max_tokens_diff*
          -- one sentence is *max_tokens_ratio* times longer than the other
        """

        if len(src_ids) != len(tgt_ids):
            raise ValueError("Source and target corpora have different lengths!")
        src_ids_, tgt_ids_ = [], []
        for i in range(len(src_ids)):
            src_len, tgt_len = len(src_ids[i]), len(tgt_ids[i])
            if (
                src_len > max_tokens
                or tgt_len > max_tokens
                or src_len < min_tokens
                or tgt_len < min_tokens
                or (src_ids[i] == tgt_ids[i])
                or np.abs(src_len - tgt_len) > max_tokens_diff
            ):
                continue
            ratio = max(src_len - 2, 1) / max(tgt_len - 2, 1)
            if ratio > max_tokens_ratio or ratio < (1 / max_tokens_ratio):
                continue
            src_ids_.append(src_ids[i])
            tgt_ids_.append(tgt_ids[i])
        return src_ids_, tgt_ids_
