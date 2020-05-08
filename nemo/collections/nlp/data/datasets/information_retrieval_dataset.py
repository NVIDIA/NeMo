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

__all__ = ['BertInformationRetrievalDataset', 'BertInformationRetrievalDatasetEval']


class BertInformationRetrievalDataset(Dataset):
    def __init__(self, tokenizer, documents, queries, triples,
                 max_seq_length=256, num_negatives=5):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_negatives = num_negatives
        self.documents = self.parse_collection(documents)
        self.queries = self.parse_collection(queries)
        self.idx2triples = self.parse_triples(triples)

    def __getitem__(self, idx):
        return self.preprocess_query_and_docs(self.idx2triples[idx])

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
        return int(id_), token_ids[:self.max_seq_length]

    def parse_triples(self, file):
        idx2triples = {}
        idx = 0
        for line in open(file, "r").readlines():
            query_and_docs = line.split("\t")
            query_and_docs_ids = [int(id_) for id_ in query_and_docs]
            query_and_rel_doc_ids, irrel_docs_ids = query_and_docs_ids[:2], query_and_docs_ids[2:]
            random.shuffle(irrel_docs_ids)
            num_samples = len(irrel_docs_ids) // self.num_negatives
            for j in range(num_samples):
                left = self.num_negatives * j
                right = self.num_negatives * (j + 1)
                idx2triples[idx] = query_and_rel_doc_ids + irrel_docs_ids[left:right]
                idx += 1
        return idx2triples
    
    def preprocess_query_and_docs(self, query_and_docs):
        query_id , doc_ids = query_and_docs[0], query_and_docs[1:]
        input_ids, input_mask, input_type_ids = [], [], []
        for doc_id in doc_ids:
            input_ids_, input_mask_, input_type_ids_ = self.pair_query_doc(query_id, doc_id)
            input_ids.append(input_ids_)
            input_mask.append(input_mask_)
            input_type_ids.append(input_type_ids_)
        input_ids = np.stack(input_ids)
        input_mask = np.stack(input_mask)
        input_type_ids = np.stack(input_type_ids)
        return input_ids, input_mask, input_type_ids

    def pair_query_doc(self, query_id, doc_id):
        query_token_ids = self.queries[query_id]
        doc_token_ids = self.documents[doc_id]
        input_ids = [self.tokenizer.pad_id] * self.max_seq_length
        bert_input = [self.tokenizer.cls_id] + query_token_ids + [self.tokenizer.sep_id]
        sentence_a_length = len(bert_input)
        bert_input = bert_input + doc_token_ids + [self.tokenizer.sep_id]

        if len(bert_input) >= self.max_seq_length:
            bert_input = bert_input[:self.max_seq_length-1] + [self.tokenizer.sep_id]

        num_nonpad_tokens = len(bert_input)
        input_ids[:num_nonpad_tokens] = bert_input
        input_ids = np.array(input_ids, dtype=np.long)
        input_mask = (input_ids != self.tokenizer.pad_id)
        input_type_ids = np.ones_like(input_ids)
        input_type_ids[:sentence_a_length] = 0
        
        return input_ids, input_mask, input_type_ids


class BertInformationRetrievalDatasetEval(Dataset):
    def __init__(self, tokenizer, documents, queries, qrels, topk_list,
                 max_seq_length=256, num_candidates=10):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_candidates = num_candidates
        self.documents = self.parse_collection(documents)
        self.queries = self.parse_collection(queries)
        self.query2rel = self.parse_qrels(qrels)
        self.idx2topk = self.parse_topk_list(topk_list)

    def __getitem__(self, idx):
        return self.preprocess_query_and_docs(self.idx2topk[idx])

    def __len__(self):
        return len(self.idx2topk)
    
    def parse_qrels(self, qrels):
        query2rel = {}
        for line in open(qrels, "r").readlines():
            query_id = int(line.split("\t")[0])
            doc_id = int(line.split("\t")[2])
            if query_id not in query2rel:
                query2rel[query_id] = [doc_id]
            else:
                query2rel[query_id].append(doc_id)
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
        return int(id_), token_ids[:self.max_seq_length]

    def parse_topk_list(self, file):
        idx2topk = {}
        for i, line in enumerate(open(file, "r").readlines()):
            query_and_docs = line.split("\t")[:self.num_candidates+1]
            idx2topk[i] = [int(id_) for id_ in query_and_docs]
        return idx2topk
    
    def preprocess_query_and_docs(self, query_and_docs):
        query_id , doc_ids = query_and_docs[0], query_and_docs[1:]
        input_ids, input_mask, input_type_ids, doc_rels = [], [], [], []
        for doc_id in doc_ids:
            input_ids_, input_mask_, input_type_ids_ = self.pair_query_doc(query_id, doc_id)
            input_ids.append(input_ids_)
            input_mask.append(input_mask_)
            input_type_ids.append(input_type_ids_)
            doc_rels.append(doc_id in self.query2rel[query_id])
        input_ids = np.stack(input_ids)
        input_mask = np.stack(input_mask)
        input_type_ids = np.stack(input_type_ids)
        doc_rels = np.array(doc_rels, dtype=np.long)
        return input_ids, input_mask, input_type_ids, doc_rels

    def pair_query_doc(self, query_id, doc_id):
        query_token_ids = self.queries[query_id]
        doc_token_ids = self.documents[doc_id]
        input_ids = [self.tokenizer.pad_id] * self.max_seq_length
        bert_input = [self.tokenizer.cls_id] + query_token_ids + [self.tokenizer.sep_id]
        sentence_a_length = len(bert_input)
        bert_input = bert_input + doc_token_ids + [self.tokenizer.sep_id]

        if len(bert_input) >= self.max_seq_length:
            bert_input = bert_input[:self.max_seq_length-1] + [self.tokenizer.sep_id]

        num_nonpad_tokens = len(bert_input)
        input_ids[:num_nonpad_tokens] = bert_input
        input_ids = np.array(input_ids, dtype=np.long)
        input_mask = (input_ids != self.tokenizer.pad_id)
        input_type_ids = np.ones_like(input_ids)
        input_type_ids[:sentence_a_length] = 0
        
        return input_ids, input_mask, input_type_ids
