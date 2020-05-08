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

import os
import pickle
import random
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset

__all__ = ['BertDensePassageRetrievalDataset', 'BertDensePassageRetrievalDatasetEval']


class BertDensePassageRetrievalDataset(Dataset):
    def __init__(self, tokenizer, passages, queries, triples,
                 max_query_length=32, max_passage_length=192, num_negatives=1):
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length
        self.num_negatives = num_negatives
        self.passages = self.parse_collection(passages)
        self.queries = self.parse_collection(queries)
        self.idx2triples = self.parse_triples(triples)

    def __getitem__(self, idx):
        return self.preprocess_query_and_passages(self.idx2triples[idx])

    def __len__(self):
        return len(self.idx2triples)

    def parse_collection(self, file):
        cached_collection = file + ".pkl"
        if os.path.isfile(cached_collection):
            file_dict = pickle.load(open(cached_collection, "rb"))
        else:
            file_dict = {}
            lines = open(file, "r").readlines()
            with mp.Pool() as pool:
                file_dict = pool.map(self.preprocess_line, lines)
            file_dict = {q[0]:q[1] for q in file_dict}
            pickle.dump(file_dict, open(cached_collection, "wb"))
        return file_dict

    def preprocess_line(self, line):
        id_, text = line.split("\t")
        token_ids = self.tokenizer.text_to_ids(text.strip())
        return int(id_), token_ids[:self.max_passage_length]

    def parse_triples(self, file):
        idx2triples = {}
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
                idx2triples[idx] = query_and_rel_psg_ids + irrel_psgs_ids[left:right]
                idx += 1
        return idx2triples

    def preprocess_query_and_passages(self, query_and_passages):
        query_id, passage_ids = query_and_passages[0], query_and_passages[1:]

        q_ids, q_mask, q_type_ids = self.prepare_input(
            token_ids_list=[self.queries[query_id]],
            max_length=self.max_query_length)

        p_ids, p_mask, p_type_ids = self.prepare_input(
            token_ids_list=[self.passages[psg_id] for psg_id in passage_ids],
            max_length=self.max_passage_length)
        return q_ids, q_mask, q_type_ids, p_ids, p_mask, p_type_ids

    def prepare_input(self, token_ids_list, max_length):
        input_ids_list, input_mask_list, input_type_ids_list = [], [], []
        for token_ids in token_ids_list:
            input_ids = [self.tokenizer.pad_id] * max_length
            bert_input = [self.tokenizer.cls_id] + token_ids[:max_length-2] + [self.tokenizer.sep_id]
            num_nonpad_tokens = len(bert_input)
            input_ids[:num_nonpad_tokens] = bert_input
            input_ids = np.array(input_ids, dtype=np.long)
            input_mask = (input_ids != self.tokenizer.pad_id)
            input_type_ids = np.zeros_like(input_ids)
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            input_type_ids_list.append(input_type_ids)
        input_ids = np.stack(input_ids_list)
        input_mask = np.stack(input_mask_list)
        input_type_ids = np.stack(input_type_ids_list)
        return input_ids, input_mask, input_type_ids


class BertDensePassageRetrievalDatasetEval(Dataset):
    def __init__(self, tokenizer, passages, queries, qrels, topk_list,
                 max_query_length=32, max_passage_length=192, num_candidates=10):
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length
        self.num_candidates = num_candidates
        self.passages = self.parse_collection(passages)
        self.queries = self.parse_collection(queries)
        self.query2rel = self.parse_qrels(qrels)
        self.idx2topk = self.parse_topk_list(topk_list)

    def __getitem__(self, idx):
        return self.preprocess_query_and_passages(self.idx2topk[idx])

    def __len__(self):
        return len(self.idx2topk)

    def parse_qrels(self, qrels):
        query2rel = {}
        for line in open(qrels, "r").readlines():
            query_id = int(line.split("\t")[0])
            psg_id = int(line.split("\t")[2])
            if query_id not in query2rel:
                query2rel[query_id] = [psg_id]
            else:
                query2rel[query_id].append(psg_id)
        return query2rel

    def parse_collection(self, file):
        cached_collection = file + ".pkl"
        if os.path.isfile(cached_collection):
            file_dict = pickle.load(open(cached_collection, "rb"))
        else:
            file_dict = {}
            lines = open(file, "r").readlines()
            with mp.Pool() as pool:
                file_dict = pool.map(self.preprocess_line, lines)
            file_dict = {q[0]:q[1] for q in file_dict}
            pickle.dump(file_dict, open(cached_collection, "wb"))
        return file_dict

    def preprocess_line(self, line):
        id_, text = line.split("\t")
        token_ids = self.tokenizer.text_to_ids(text.strip())
        return int(id_), token_ids[:self.max_passage_length]

    def parse_topk_list(self, file):
        idx2topk = {}
        for i, line in enumerate(open(file, "r").readlines()):
            query_and_passages = line.split("\t")[:self.num_candidates+1]
            idx2topk[i] = [int(id_) for id_ in query_and_passages]
        return idx2topk

    def preprocess_query_and_passages(self, query_and_passages):
        query_id, passage_ids = query_and_passages[0], query_and_passages[1:]

        q_ids, q_mask, q_type_ids = self.prepare_input(
            token_ids_list=[self.queries[query_id]],
            max_length=self.max_query_length)

        p_ids, p_mask, p_type_ids = self.prepare_input(
            token_ids_list=[self.passages[psg_id] for psg_id in passage_ids],
            max_length=self.max_passage_length)

        psg_rels = [(psg_id in self.query2rel[query_id]) for psg_id in passage_ids]
        psg_rels = np.array(psg_rels, dtype=np.long)

        return q_ids, q_mask, q_type_ids, p_ids, p_mask, p_type_ids, psg_rels

    def prepare_input(self, token_ids_list, max_length):
        input_ids_list, input_mask_list, input_type_ids_list = [], [], []
        for token_ids in token_ids_list:
            input_ids = [self.tokenizer.pad_id] * max_length
            bert_input = [self.tokenizer.cls_id] + token_ids[:max_length-2] + [self.tokenizer.sep_id]
            num_nonpad_tokens = len(bert_input)
            input_ids[:num_nonpad_tokens] = bert_input
            input_ids = np.array(input_ids, dtype=np.long)
            input_mask = (input_ids != self.tokenizer.pad_id)
            input_type_ids = np.zeros_like(input_ids)
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            input_type_ids_list.append(input_type_ids)
        input_ids = np.stack(input_ids_list)
        input_mask = np.stack(input_mask_list)
        input_type_ids = np.stack(input_type_ids_list)
        return input_ids, input_mask, input_type_ids