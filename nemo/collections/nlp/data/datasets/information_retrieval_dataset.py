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
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset

__all__ = ['BertInformationRetrievalDataset']


class BertInformationRetrievalDataset(Dataset):
    def __init__(self, tokenizer, documents, queries, qrels, max_seq_length=256):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.documents = self.parse_collection(documents)
        self.queries = self.parse_collection(queries)
        self.idx2query = self.parse_qrels(qrels)
        
    def __getitem__(self, idx):
        query_id, rel_doc_id, irrel_doc_id = self.idx2query[idx]
        rel_input_ids, rel_mask, rel_type_ids = self.pair_query_doc(query_id, rel_doc_id)
        irrel_input_ids, irrel_mask, irrel_type_ids = self.pair_query_doc(query_id, irrel_doc_id)
        return rel_input_ids, rel_mask, rel_type_ids, irrel_input_ids, irrel_mask, irrel_type_ids

#     def __getitem__(self, idx):
#         query_id, rel_doc_id, irrel_doc_id = self.idx2query[idx]
#         rel_input_ids, rel_mask, rel_type_ids, rel_nonpad = \
#             self.pair_query_doc(query_id, rel_doc_id)
#         irrel_input_ids, irrel_mask, irrel_type_ids, irrel_nonpad = \
#             self.pair_query_doc(query_id, irrel_doc_id)
#         return rel_input_ids, rel_mask, rel_type_ids, rel_nonpad, \
#             irrel_input_ids, irrel_mask, irrel_type_ids, irrel_nonpad

    def __len__(self):
        return len(self.idx2query)

    def parse_collection(self, file):
        cached_collection = file + ".pkl"
        if os.path.isfile(cached_collection):
            file_dict = pickle.load(open(cached_collection, "rb"))
        else:
            file_dict = {}
            lines = open(file, "r").readlines()
            with mp.Pool(4) as pool:
                file_dict = pool.map(self.preprocess_line, lines)
            file_dict = {q[0]:q[1] for q in file_dict}
            pickle.dump(file_dict, open(cached_collection, "wb"))
        return file_dict

    def preprocess_line(self, line):
        id_, text = line.split("\t")
        token_ids = self.tokenizer.text_to_ids(text.strip())
        return int(id_), token_ids[:self.max_seq_length]

    def parse_qrels(self, file):
        idx2query = {}
        for i, line in enumerate(open(file, "r").readlines()):
            idx2query[i] = [int(id_) for id_ in line.split("\t")][:3]
        return idx2query

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
        
        return input_ids, input_mask, input_type_ids#, num_nonpad_tokens
