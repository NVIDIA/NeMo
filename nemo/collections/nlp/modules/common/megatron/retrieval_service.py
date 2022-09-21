# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import multiprocessing
import pathlib
import sys
import time
from multiprocessing import Pool
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

import faiss
import numpy as np
import torch
from numba import njit, prange
from sentence_transformers import SentenceTransformer
from typing import Union, List

from nemo.collections.nlp.data.language_modeling.megatron.indexed_retrieval_dataset import (
    KNNIndex,
    MMapRetrievalIndexedDataset,
    merge_knn_files,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.utils import logging
import abc


class RetrievalService:

    @abc.abstractmethod
    def get_knn(self, query: Union[List[str], str], neighbors: int):
        pass

def calculate_embedding(pool, batch_size):
    while True:
        sentences, slice_id = get_sentence_chunks()
        if sentences is None:
            break
        beg = time.time()
        emb = model.encode_multi_process(sentences=sentences, pool=pool, batch_size=batch_size)
        end = time.time()
        logging.info(f"one embedding {len(emb)} batch size takes {end-beg}")
        emb_queue.put((emb, slice_id))
    emb_queue.put((None, None))


class FaissRetrievalService(RetrievalService):

    def __init__(self,
                 faiss_index: str,
                 faiss_devices: str,
                 nprobe: int,
                 retrieval_index: str,
                 tokenizer: TokenizerSpec,
                 sentence_bert: str = 'all-mpnet-base-v2',
                 sentence_bert_batch: int = 4):
        has_gpu = torch.cuda.is_available() and hasattr(faiss, "index_gpu_to_cpu")

        if faiss_devices is None or not torch.cuda.is_available():
            device_list = None
        else:
            device_list = ['cuda:' + str(device) for device in faiss_devices.split(',')]

        index = faiss.read_index(faiss_index)
        if has_gpu and device_list is not None:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.usePrecomputed = False
            co.shard = True
            index = faiss.index_cpu_to_all_gpus(index, co, ngpu=len(device_list))
        index.nprobe = nprobe

        self.bert_model = SentenceTransformer(sentence_bert)
        self.tokenizer = tokenizer
        self.ds = MMapRetrievalIndexedDataset(retrieval_index)
        self.pool = self.bert_model.start_multi_process_pool(device_list)
        self.sentence_bert_batch = sentence_bert_batch

    def get_knn(self, query: Union[List[str], str], neighbors: int):
        if isinstance(query, str):
            query = [query]
        emb = self.bert_model.encode_multi_process(sentences=sentences, pool=self.pool, batch_size=self.sentence_bert_batch)
        D, I = index.search(emb, neighbors)
        return I
