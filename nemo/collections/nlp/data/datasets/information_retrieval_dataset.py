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

"""Pytorch Dataset for training Information Retrieval."""

import multiprocessing as mp
import os
import pickle
import random

import numpy as np
from torch.utils.data import Dataset

__all__ = [
    "BertInformationRetrievalDatasetTrain",
    "BertInformationRetrievalDatasetEval",
    "BertDensePassageRetrievalDatasetInfer",
]


class BaseInformationRetrievalDataset(Dataset):
    def __init__(self, tokenizer, max_query_length=31, max_passage_length=190):
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length

    def parse_npz(self, file, max_seq_length):
        cached_collection = file + ".npz"
        if os.path.isfile(cached_collection):
            file_npz = np.load(cached_collection)["data"]
        else:
            file_dict = {}
            lines = open(file, "r").readlines()
            with mp.Pool() as pool:
                file_dict = pool.map(self.preprocess_line, lines)
            file_dict = {q[0]: q[1] for q in file_dict}
            file_npz = np.zeros((len(file_dict), max_seq_lenth))
            for key in file_dict:
                file_npz[key][0] = len(file_dict[key])
                file_npz[key][1 : len(file_dict[key]) + 1] = file_dict[key]
            np.savez(cached_collection, data=file_npz)
        return file_npz

    def parse_pkl(self, file, max_seq_length):
        cached_collection = file + ".pkl"
        if os.path.isfile(cached_collection):
            file_dict = pickle.load(open(cached_collection, "rb"))
        else:
            file_dict = {}
            lines = open(file, "r").readlines()
            with mp.Pool() as pool:
                file_dict = pool.map(self.preprocess_line, lines)
            file_dict = {q[0]: q[1] for q in file_dict}
            pickle.dump(file_dict, open(cached_collection, "wb"))
        return {key: file_dict[key][:max_seq_length] for key in file_dict}

    def preprocess_line(self, line):
        id_, text = line.split("\t")
        token_ids = self.tokenizer.text_to_ids(text.strip())
        return int(id_), token_ids[: self.max_passage_length]

    def construct_input(self, token_ids1, max_seq_length, token_ids2=None):
        input_ids = [self.tokenizer.pad_id] * max_seq_length
        bert_input = [self.tokenizer.cls_id] + token_ids1 + [self.tokenizer.sep_id]
        sentence1_length = len(bert_input)
        if token_ids2 is not None:
            bert_input = bert_input + token_ids2 + [self.tokenizer.sep_id]
        num_nonpad_tokens = len(bert_input)

        input_ids[:num_nonpad_tokens] = bert_input
        input_ids = np.array(input_ids, dtype=np.long)
        input_mask = input_ids != self.tokenizer.pad_id
        input_type_ids = np.ones_like(input_ids)
        input_type_ids[:sentence1_length] = 0

        return input_ids, input_mask, input_type_ids

    def preprocess_bert(self, query_id, psg_ids):
        max_seq_length = self.max_query_length + self.max_passage_length + 3
        input_ids, input_mask, input_type_ids = [], [], []
        for psg_id in psg_ids:
            inputs = self.construct_input(self.queries[query_id], max_seq_length, self.psgid2tokens(psg_id))
            input_ids.append(inputs[0])
            input_mask.append(inputs[1])
            input_type_ids.append(inputs[2])
        input_ids = np.stack(input_ids)
        input_mask = np.stack(input_mask)
        input_type_ids = np.stack(input_type_ids)
        return input_ids, input_mask, input_type_ids

    def preprocess_dpr(self, query_id, psg_ids):
        q_input_ids, q_input_mask, q_type_ids = self.construct_input(self.queries[query_id], self.max_query_length + 2)
        input_ids, input_mask, input_type_ids = [], [], []
        for psg_id in psg_ids:
            inputs = self.construct_input(self.psgid2tokens(psg_id), self.max_passage_length + 2)
            input_ids.append(inputs[0])
            input_mask.append(inputs[1])
            input_type_ids.append(inputs[2])
        input_ids = np.stack(input_ids)
        input_mask = np.stack(input_mask)
        input_type_ids = np.stack(input_type_ids)
        return (
            q_input_ids[None, ...],
            q_input_mask[None, ...],
            q_type_ids[None, ...],
            input_ids,
            input_mask,
            input_type_ids,
        )

    def psgid2tokens(self, psg_id):
        seq_len = self.passages[psg_id][0]
        return self.passages[psg_id][1 : seq_len + 1].tolist()


class BertInformationRetrievalDatasetTrain(BaseInformationRetrievalDataset):
    def __init__(
        self,
        tokenizer,
        passages,
        queries,
        query_to_passages,
        max_query_length=31,
        max_passage_length=190,
        num_negatives=10,
        preprocess_fn="preprocess_bert",
    ):
        super().__init__(tokenizer, max_query_length, max_passage_length)
        self.num_negatives = num_negatives

        self.passages = self.parse_npz(passages, max_passage_length)
        self.queries = self.parse_pkl(queries, max_query_length)
        self.idx2psgs = self.parse_query_to_passages(query_to_passages)
        self._preprocess_fn = getattr(self, preprocess_fn)

    def __getitem__(self, idx):
        query_and_psgs = self.idx2psgs[idx]
        query_id, psg_ids = query_and_psgs[0], query_and_psgs[1:]
        inputs = self._preprocess_fn(query_id, psg_ids)
        return inputs

    def __len__(self):
        return len(self.idx2psgs)

    def parse_query_to_passages(self, file):
        idx2psgs = {}
        idx = 0
        for line in open(file, "r").readlines():
            query_and_psgs = line.split("\t")
            query_and_psgs_ids = [int(id_) for id_ in query_and_psgs]
            query_and_rel_psg_ids, irrel_psgs_ids = query_and_psgs_ids[:2], query_and_psgs_ids[2:]
            random.shuffle(irrel_psgs_ids)
            num_samples = len(irrel_psgs_ids) // self.num_negatives
            for j in range(num_samples):
                left = self.num_negatives * j
                right = self.num_negatives * (j + 1)
                idx2psgs[idx] = query_and_rel_psg_ids + irrel_psgs_ids[left:right]
                idx += 1
        return idx2psgs


class BertInformationRetrievalDatasetEval(BaseInformationRetrievalDataset):
    def __init__(
        self,
        tokenizer,
        passages,
        queries,
        query_to_passages,
        max_query_length=31,
        max_passage_length=190,
        num_candidates=10,
        preprocess_fn="preprocess_bert",
    ):
        super().__init__(tokenizer, max_query_length, max_passage_length)
        self.num_candidates = num_candidates

        self.passages = self.parse_npz(passages, max_passage_length)
        self.queries = self.parse_pkl(queries, max_query_length)
        self.idx2topk = self.parse_topk_list(query_to_passages)
        self._preprocess_fn = getattr(self, preprocess_fn)

    def __getitem__(self, idx):
        query_and_psgs = self.idx2topk[idx]
        query_id, psg_ids = query_and_psgs[0], query_and_psgs[1:]
        inputs = self._preprocess_fn(query_id, psg_ids)
        return [*inputs, query_id, np.array(psg_ids)]

    def __len__(self):
        return len(self.idx2topk)

    def parse_topk_list(self, file):
        idx2topk = {}
        idx = 0
        for line in open(file, "r").readlines():
            query_and_psgs = [int(id_) for id_ in line.split("\t")]
            num_samples = int(np.ceil((len(query_and_psgs) - 1) / self.num_candidates))
            for j in range(num_samples):
                left = self.num_candidates * j + 1
                right = self.num_candidates * (j + 1) + 1
                idx2topk[idx] = [query_and_psgs[0]] + query_and_psgs[left:right]
                idx += 1
        return idx2topk


class BertDensePassageRetrievalDatasetInfer(BaseInformationRetrievalDataset):
    def __init__(self, tokenizer, passages=None, queries=None, max_query_length=31, max_passage_length=190):
        super().__init__(tokenizer, max_query_length, max_passage_length)

        if passages is not None:
            self.passages = self.parse_npz(passages, max_passage_length)
            self.max_seq_length = max_passage_length
            self._get_tokens = self.psgid2tokens
            self.idx2dataid = {psg_id: psg_id for psg_id in range(self.passages.shape[0])}
        elif queries is not None:
            self.queries = self.parse_pkl(queries, max_query_length)
            self.max_seq_length = max_query_length
            self._get_tokens = self.queryid2tokens
            self.idx2dataid = {i: query_id for i, query_id in enumerate(self.queries)}

    def __getitem__(self, idx):
        data_id = self.idx2dataid[idx]
        token_ids = self._get_tokens(data_id)
        inputs = self.construct_input(token_ids, self.max_seq_length + 2)
        return [*inputs, data_id]

    def __len__(self):
        return len(self.idx2dataid)

    def queryid2tokens(self, query_id):
        return self.queries[query_id]
