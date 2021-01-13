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

"""Pytorch Dataset for training Neural Machine Translation."""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from omegaconf.omegaconf import MISSING

from nemo.collections.nlp.data.data_utils.data_preprocessing import dataset_to_ids
from nemo.core import Dataset

__all__ = ['TranslationDataset']


@dataclass
class TranslationDataConfig:
    src_file_name: str = None  # MISSING
    tgt_file_name: str = None  # MISSING
    tokens_in_batch: int = 512
    clean: bool = False
    max_seq_length: int = 512
    cache_ids: bool = False
    cache_data_per_node: bool = False
    use_cache: bool = False
    shuffle: bool = False
    num_samples: int = -1
    drop_last: bool = False
    pin_memory: bool = False
    num_workers: int = 8
    load_from_cached_dataset: bool = False
    reverse_lang_direction: bool = False


class TranslationDataset(Dataset):
    def __init__(
        self,
        dataset_src,
        dataset_tgt,
        tokens_in_batch=1024,
        clean=False,
        max_seq_length=512,
        min_seq_length=1,
        max_seq_length_diff=512,
        max_seq_length_ratio=512,
        cache_ids=False,
        cache_data_per_node=False,
        use_cache=False,
        reverse_lang_direction=False,
    ):
        self.dataset_src = dataset_src
        self.dataset_tgt = dataset_tgt
        self.tokens_in_batch = tokens_in_batch
        self.cache_ids = cache_ids
        self.use_cache = use_cache
        self.clean = clean
        self.cache_data_per_node = cache_data_per_node
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.max_seq_length_diff = max_seq_length_diff
        self.max_seq_length_ratio = max_seq_length_ratio
        self.reverse_lang_direction = reverse_lang_direction

    def batchify(self, tokenizer_src, tokenizer_tgt):
        src_ids = dataset_to_ids(
            self.dataset_src,
            tokenizer_src,
            cache_ids=self.cache_ids,
            cache_data_per_node=self.cache_data_per_node,
            use_cache=self.use_cache,
        )
        tgt_ids = dataset_to_ids(
            self.dataset_tgt,
            tokenizer_tgt,
            cache_ids=self.cache_ids,
            cache_data_per_node=self.cache_data_per_node,
            use_cache=self.use_cache,
        )
        if self.clean:
            src_ids, tgt_ids = self.clean_src_and_target(
                src_ids,
                tgt_ids,
                max_tokens=self.max_seq_length,
                min_tokens=self.min_seq_length,
                max_tokens_diff=self.max_seq_length_diff,
                max_tokens_ratio=self.max_seq_length_ratio,
            )
        self.src_pad_id = tokenizer_src.pad_id
        self.tgt_pad_id = tokenizer_tgt.pad_id

        self.batch_indices = self.pack_data_into_batches(src_ids, tgt_ids)
        self.batches = self.pad_batches(src_ids, tgt_ids, self.batch_indices)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        src_ids = self.batches[idx]["src"]
        tgt = self.batches[idx]["tgt"]
        if self.reverse_lang_direction:
            src_ids, tgt = tgt, src_ids
        labels = tgt[:, 1:]
        tgt_ids = tgt[:, :-1]
        src_mask = (src_ids != self.src_pad_id).astype(np.int32)
        tgt_mask = (tgt_ids != self.tgt_pad_id).astype(np.int32)
        sent_ids = np.array(self.batch_indices[idx])
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
            src_ids_ = self.src_pad_id * np.ones((len(b), src_len), dtype=np.int)
            tgt_ids_ = self.tgt_pad_id * np.ones((len(b), tgt_len), dtype=np.int)
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
            if src_len not in buckets:
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

        while i < len(buckets):
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
        self,
        src_ids,
        tgt_ids,
        max_tokens=None,
        min_tokens=None,
        max_tokens_diff=None,
        max_tokens_ratio=None,
        filter_equal_src_and_dest=False,
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
                (max_tokens is not None and (src_len > max_tokens or tgt_len > max_tokens))
                or (min_tokens is not None and (src_len < min_tokens or tgt_len < min_tokens))
                or (filter_equal_src_and_dest and src_ids[i] == tgt_ids[i])
                or (max_tokens_diff is not None and np.abs(src_len - tgt_len) > max_tokens_diff)
            ):
                continue
            if max_tokens_ratio is not None:
                ratio = max(src_len - 2, 1) / max(tgt_len - 2, 1)
                if ratio > max_tokens_ratio or ratio < (1 / max_tokens_ratio):
                    continue
            src_ids_.append(src_ids[i])
            tgt_ids_.append(tgt_ids[i])
        return src_ids_, tgt_ids_
