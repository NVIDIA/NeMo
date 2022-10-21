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

import abc
import json
import logging
import threading
import time
from typing import List, Union

import faiss
import numpy as np
import requests
import torch
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from sentence_transformers import SentenceTransformer

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.indexed_retrieval_dataset import MMapRetrievalIndexedDataset

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

lock = threading.Lock()
headers = {"Content-Type": "application/json"}

PORT_NUM = 17179


class RetrievalService:
    @abc.abstractmethod
    def get_knn(self, query: Union[List[str], str, torch.Tensor], neighbors: int):
        pass


class FaissRetrievalResource(Resource):
    def __init__(
        self, index, bert_model, tokenizer, ds, pool, sentence_bert_batch,
    ):
        # server
        self.index = index
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.ds = ds
        self.pool = pool
        self.sentence_bert_batch = sentence_bert_batch

    def put(self):
        # logging.info("request IP: " + str(request.remote_addr))
        # logging.info(json.dumps(request.get_json()))
        data = request.get_json()
        sentences = data['sentences']
        num_neighbors = data['neighbors']
        with lock:  # Need to get lock to keep multiple threads from hitting code
            neighbors = self.get_knn(sentences, num_neighbors)
        return jsonify(neighbors.tolist())
        # check keys

    def get_knn(self, query: Union[List[str], str, torch.Tensor], neighbors: int):
        single_sentence = False
        if isinstance(query, str):
            single_sentence = True
            query = [query]
        elif isinstance(query, torch.Tensor):
            sentence_list = []
            for q in query:
                text = self.tokenizer.ids_to_text(q)
                sentence_list.append(text)
            query = sentence_list
        emb = self.bert_model.encode_multi_process(
            sentences=query, pool=self.pool, batch_size=self.sentence_bert_batch
        )
        D, knn = self.index.search(emb, neighbors)
        results = []
        for sentence_neighbors in knn:
            chunks = []
            for neighbor_chunk_id in sentence_neighbors:
                chunk_id = self.ds.get_chunk(neighbor_chunk_id)
                chunks.append(chunk_id)
            chunks = np.stack(chunks, axis=0).astype(np.int64)
            results.append(chunks)
        if single_sentence:
            # unpack the single sentence input
            return results[0]
        return np.stack(results, axis=0).astype(np.int64)


class RetrievalServer(object):
    def __init__(
        self,
        faiss_index: str,
        faiss_devices: str,
        nprobe: int,
        retrieval_index: str,
        tokenizer: TokenizerSpec,
        sentence_bert: str = 'all-mpnet-base-v2',
        sentence_bert_batch: int = 4,
    ):
        self.app = Flask(__name__, static_url_path='')
        # server
        has_gpu = torch.cuda.is_available() and hasattr(faiss, "index_gpu_to_cpu")

        if faiss_devices is None or not torch.cuda.is_available():
            device_list = None
        else:
            device_list = ['cuda:' + str(device) for device in faiss_devices.split(',')]

        self.index = faiss.read_index(faiss_index)
        if has_gpu and device_list is not None:
            beg = time.time()
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.usePrecomputed = False
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co, ngpu=len(device_list))
            end = time.time()
            logging.info('convert Faiss db to GPU takes', end - beg)
        self.index.nprobe = nprobe
        self.bert_model = SentenceTransformer(sentence_bert)
        self.tokenizer = tokenizer
        self.ds = MMapRetrievalIndexedDataset(retrieval_index)
        self.pool = self.bert_model.start_multi_process_pool(device_list)
        self.sentence_bert_batch = sentence_bert_batch
        api = Api(self.app)
        api.add_resource(
            FaissRetrievalResource,
            '/knn',
            resource_class_args=[
                self.index,
                self.bert_model,
                self.tokenizer,
                self.ds,
                self.pool,
                self.sentence_bert_batch,
            ],
        )

    def run(self, url, port=PORT_NUM):
        threading.Thread(target=lambda: self.app.run(host=url, threaded=True, port=port)).start()


# class OnTheFlyFaissRetrievalService(RetrievalService):
#     def __init__(
#         self,
#         faiss_devices: str,
#         embedding_dim: int,
#         tokenizer: TokenizerSpec,
#         sentence_bert: str = 'all-mpnet-base-v2',
#         sentence_bert_batch: int = 4,
#         neighbors: int = 4,
#     ):
#         self.neighbors = neighbors
#         has_gpu = torch.cuda.is_available() and hasattr(faiss, "index_gpu_to_cpu")
#         self.index = faiss.IndexFlatL2(embedding_dim)   # build the index
#
#         if faiss_devices is None or not torch.cuda.is_available():
#             device_list = None
#         else:
#             device_list = ['cuda:' + str(device) for device in faiss_devices.split(',')]
#
#         if has_gpu and device_list is not None:
#             beg = time.time()
#             co = faiss.GpuMultipleClonerOptions()
#             co.useFloat16 = True
#             co.usePrecomputed = False
#             co.shard = True
#             self.index = faiss.index_cpu_to_all_gpus(self.index, co, ngpu=len(device_list))
#             end = time.time()
#             logging.info('convert Faiss db to GPU takes', end - beg)
#         self.index.nprobe = nprobe
#
#         self.bert_model = SentenceTransformer(sentence_bert)
#         self.tokenizer = tokenizer
#         self.ds = MMapRetrievalIndexedDataset(retrieval_index)
#         self.pool = self.bert_model.start_multi_process_pool(device_list)
#         self.sentence_bert_batch = sentence_bert_batch
#
#     def update_index(self, content)
#         self.index.add(xb)                  # add vectors to the index
#         print(index.ntotal)
#
#     def get_knn(self, query: Union[List[str], str, torch.Tensor]):
#         single_sentence = False
#         if isinstance(query, str):
#             single_sentence = True
#             query = [query]
#         elif isinstance(query, torch.Tensor):
#             sentence_list = []
#             for q in query:
#                 text = self.tokenizer.ids_to_text(q)
#                 sentence_list.append(text)
#             query = sentence_list
#         emb = self.bert_model.encode_multi_process(
#             sentences=query, pool=self.pool, batch_size=self.sentence_bert_batch
#         )
#         D, knn = self.index.search(emb, self.neighbors)
#         results = []
#         for sentence_neighbors in knn:
#             chunks = []
#             for neighbor_chunk_id in sentence_neighbors:
#                 chunk_id = self.ds.get_chunk(neighbor_chunk_id)
#                 chunks.append(chunk_id)
#             chunks = np.stack(chunks, axis=0).astype(np.int64)
#             results.append(chunks)
#         if single_sentence:
#             # unpack the single sentence input
#             return results[0]
#         return np.stack(results, axis=0).astype(np.int64)
#


def request_data(data):
    resp = requests.put('http://localhost:{}/knn'.format(PORT_NUM), data=json.dumps(data), headers=headers)
    return resp.json()


class FaissRetrievalService(RetrievalService):
    def __init__(
        self,
        faiss_index: str,
        faiss_devices: str,
        nprobe: int,
        retrieval_index: str,
        tokenizer: TokenizerSpec,
        sentence_bert: str = 'all-mpnet-base-v2',
        sentence_bert_batch: int = 4,
    ):
        self.tokenizer = tokenizer
        ds = MMapRetrievalIndexedDataset(retrieval_index)
        self.chunk_size = ds.chunk_size
        if torch.distributed.get_rank() == 0:
            server = RetrievalServer(
                faiss_index, faiss_devices, nprobe, retrieval_index, tokenizer, sentence_bert, sentence_bert_batch,
            )
            server.run("0.0.0.0")
        torch.distributed.barrier()

    def get_knn(self, query: Union[List[str], str, torch.Tensor], neighbors):
        if isinstance(query, torch.Tensor):
            sentence_list = []
            for q in query:
                text = self.tokenizer.ids_to_text(q)
                sentence_list.append(text)
            query = sentence_list
        data = {'sentences': query}
        data['neighbors'] = neighbors
        result = request_data(data)
        result = np.array(result)
        return result


# class OnTheFlyFaissRetrievalService(RetrievalService):
#     def __init__(
#         self,
#         faiss_devices: str,
#         embedding_dim: int,
#         tokenizer: TokenizerSpec,
#         sentence_bert: str = 'all-mpnet-base-v2',
#         sentence_bert_batch: int = 4,
#         neighbors: int = 4,
#     ):
#         self.neighbors = neighbors
#         has_gpu = torch.cuda.is_available() and hasattr(faiss, "index_gpu_to_cpu")
#         self.index = faiss.IndexFlatL2(embedding_dim)   # build the index
#
#         if faiss_devices is None or not torch.cuda.is_available():
#             device_list = None
#         else:
#             device_list = ['cuda:' + str(device) for device in faiss_devices.split(',')]
#
#         if has_gpu and device_list is not None:
#             beg = time.time()
#             co = faiss.GpuMultipleClonerOptions()
#             co.useFloat16 = True
#             co.usePrecomputed = False
#             co.shard = True
#             self.index = faiss.index_cpu_to_all_gpus(self.index, co, ngpu=len(device_list))
#             end = time.time()
#             logging.info('convert Faiss db to GPU takes', end - beg)
#         self.index.nprobe = nprobe
#
#         self.bert_model = SentenceTransformer(sentence_bert)
#         self.tokenizer = tokenizer
#         self.ds = MMapRetrievalIndexedDataset(retrieval_index)
#         self.pool = self.bert_model.start_multi_process_pool(device_list)
#         self.sentence_bert_batch = sentence_bert_batch
#
#     def update_index(self, content)
#         self.index.add(xb)                  # add vectors to the index
#         print(index.ntotal)
#
#     def get_knn(self, query: Union[List[str], str, torch.Tensor]):
#         single_sentence = False
#         if isinstance(query, str):
#             single_sentence = True
#             query = [query]
#         elif isinstance(query, torch.Tensor):
#             sentence_list = []
#             for q in query:
#                 text = self.tokenizer.ids_to_text(q)
#                 sentence_list.append(text)
#             query = sentence_list
#         emb = self.bert_model.encode_multi_process(
#             sentences=query, pool=self.pool, batch_size=self.sentence_bert_batch
#         )
#         D, knn = self.index.search(emb, self.neighbors)
#         results = []
#         for sentence_neighbors in knn:
#             chunks = []
#             for neighbor_chunk_id in sentence_neighbors:
#                 chunk_id = self.ds.get_chunk(neighbor_chunk_id)
#                 chunks.append(chunk_id)
#             chunks = np.stack(chunks, axis=0).astype(np.int64)
#             results.append(chunks)
#         if single_sentence:
#             # unpack the single sentence input
#             return results[0]
#         return np.stack(results, axis=0).astype(np.int64)
#
