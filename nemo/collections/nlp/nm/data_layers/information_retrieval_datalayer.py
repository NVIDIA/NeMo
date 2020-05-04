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

import torch
from torch.utils import data as pt_data

import nemo
from nemo.collections.nlp.data import BertInformationRetrievalDataset, \
    BertInformationRetrievalDatasetMulti, BertInformationRetrievalDatasetMultiEval, \
    BertDensePassageRetrievalDataset, BertDensePassageRetrievalDatasetEval
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import ChannelType, LabelsType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = [
    'BertInformationRetrievalDataLayer',
    'BertInformationRetrievalDataLayerMulti',
    'BertInformationRetrievalDataLayerMultiEval'
]


class BertInformationRetrievalDataLayer(TextDataLayer):
    """
    Data layer for information retrieval with pre-trained Bert.

    Args:
        tokenizer (TokenizerSpec): Bert tokenizer
        documents (str): Path to the collection of documents tsv file
        queries (str): Path to the queries tsv file
        qrels (str): Path to the tsv file with query-document relevance information
        batch_size: text segments batch size
        max_seq_length (int):
                maximum allowed length of the text segments. Default: 256
        dataset_type (Dataset):
                the underlying dataset. Default: BertInformationRetrievalDataset
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        rel_ids: indices of tokens which correspond to query-relevant doc pair
        rel_mask: bool tensor with 0s in place of pad tokens in rel_ids to be masked
        irrel_ids: indices of tokens which correspond to query-irrelevant doc pair
        irrel_mask: bool tensor with 0s in place of pad tokens in irrel_ids to be masked
        """
        return {
            "rel_ids": NeuralType(('B', 'T'), ChannelType()),
            "rel_mask": NeuralType(('B', 'T'), ChannelType()),
            "rel_type_ids": NeuralType(('B', 'T'), ChannelType()),
            "irrel_ids": NeuralType(('B', 'T'), ChannelType()),
            "irrel_mask": NeuralType(('B', 'T'), ChannelType()),
            "irrel_type_ids": NeuralType(('B', 'T'), ChannelType()),
        }

    def __init__(
        self,
        tokenizer,
        documents,
        queries,
        qrels,
        batch_size,
        max_seq_length=256,
        dataset_type=BertInformationRetrievalDataset,
    ):
        dataset_params = {
            'tokenizer': tokenizer,
            'documents': documents,
            'queries': queries,
            'qrels': qrels,
            'max_seq_length': max_seq_length,
        }
        super().__init__(dataset_type, dataset_params, batch_size)


class BertInformationRetrievalDataLayerMulti(TextDataLayer):
    """
    Data layer for information retrieval with pre-trained Bert.

    Args:
        tokenizer (TokenizerSpec): Bert tokenizer
        documents (str): Path to the collection of documents tsv file
        queries (str): Path to the queries tsv file
        qrels (str): Path to the tsv file with query-document relevance information
        batch_size: text segments batch size
        max_seq_length (int):
                maximum allowed length of the text segments. Default: 256
        dataset_type (Dataset):
                the underlying dataset. Default: BertInformationRetrievalDatasetMulti
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        rel_ids: indices of tokens which correspond to query-relevant doc pair
        rel_mask: bool tensor with 0s in place of pad tokens in rel_ids to be masked
        irrel_ids: indices of tokens which correspond to query-irrelevant doc pair
        irrel_mask: bool tensor with 0s in place of pad tokens in irrel_ids to be masked
        """
        return {
            "input_ids": NeuralType(('B', 'B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'B', 'T'), ChannelType()),
            "input_type_ids": NeuralType(('B', 'B', 'T'), ChannelType()),
        }

    def __init__(
        self,
        tokenizer,
        documents,
        queries,
        triples,
        batch_size,
        max_seq_length=256,
        num_negatives=5,
        dataset_type=BertInformationRetrievalDatasetMulti,
    ):
        dataset_params = {
            'tokenizer': tokenizer,
            'documents': documents,
            'queries': queries,
            'triples': triples,
            'max_seq_length': max_seq_length,
            'num_negatives': num_negatives,
        }
        super().__init__(dataset_type, dataset_params, batch_size)


class BertInformationRetrievalDataLayerMultiEval(TextDataLayer):
    """
    Data layer for information retrieval with pre-trained Bert.

    Args:
        tokenizer (TokenizerSpec): Bert tokenizer
        documents (str): Path to the collection of documents tsv file
        queries (str): Path to the queries tsv file
        qrels (str): Path to the tsv file with query-document relevance information
        batch_size: text segments batch size
        max_seq_length (int):
                maximum allowed length of the text segments. Default: 256
        dataset_type (Dataset):
                the underlying dataset. Default: BertInformationRetrievalDatasetMulti
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        rel_ids: indices of tokens which correspond to query-relevant doc pair
        rel_mask: bool tensor with 0s in place of pad tokens in rel_ids to be masked
        irrel_ids: indices of tokens which correspond to query-irrelevant doc pair
        irrel_mask: bool tensor with 0s in place of pad tokens in irrel_ids to be masked
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), ChannelType()),
            "input_type_ids": NeuralType(('B', 'T'), ChannelType()),
            "doc_rels": NeuralType(('B', 'T'), ChannelType()),
        }

    def __init__(
        self,
        tokenizer,
        documents,
        queries,
        qrels,
        topk_list,
        max_seq_length=256,
        num_candidates=10,
        dataset_type=BertInformationRetrievalDatasetMultiEval,
    ):
        dataset_params = {
            'tokenizer': tokenizer,
            'documents': documents,
            'queries': queries,
            'qrels': qrels,
            'topk_list': topk_list,
            'max_seq_length': max_seq_length,
            'num_candidates': num_candidates,
        }
        super().__init__(dataset_type, dataset_params, batch_size=1)
        
        if self._placement == nemo.core.DeviceType.AllGpu:
            sampler = pt_data.distributed.DistributedSampler(self._dataset)
        else:
            sampler = None

        self._dataloader = pt_data.DataLoader(
            dataset=self._dataset, batch_size=1, collate_fn=self._collate_fn,
            shuffle=sampler is None, sampler=sampler
        )

    def _collate_fn(self, x):
        input_ids, input_mask, input_type_ids, doc_rels = x[0]
        input_ids = torch.Tensor(input_ids).long().to(self._device)
        input_mask = torch.Tensor(input_mask).float().to(self._device)
        input_type_ids = torch.Tensor(input_type_ids).long().to(self._device)
        doc_rels = torch.Tensor(doc_rels).long().to(self._device)
        return input_ids, input_mask, input_type_ids, doc_rels

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader


class BertDensePassageRetrievalDataLayer(TextDataLayer):
    """
    Data layer for information retrieval with pre-trained Bert.

    Args:
        tokenizer (TokenizerSpec): Bert tokenizer
        passages (str): Path to the collection of passages tsv file
        queries (str): Path to the queries tsv file
        triples (str): Path to the tsv file with query-document relevance information
        batch_size: text segments batch size
        max_query_length (int):
                maximum allowed length of the queries. Default: 32
        max_passage_length (int):
                maximum allowed length of the passages. Default: 192
        dataset_type (Dataset):
                the underlying dataset. Default: BertDensePassageRetrievalDataset
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        rel_ids: indices of tokens which correspond to query-relevant doc pair
        rel_mask: bool tensor with 0s in place of pad tokens in rel_ids to be masked
        irrel_ids: indices of tokens which correspond to query-irrelevant doc pair
        irrel_mask: bool tensor with 0s in place of pad tokens in irrel_ids to be masked
        """
        return {
            "q_input_ids": NeuralType(('B', 'B', 'T'), ChannelType()),
            "q_input_mask": NeuralType(('B', 'B', 'T'), ChannelType()),
            "q_input_type_ids": NeuralType(('B', 'B', 'T'), ChannelType()),
            "p_input_ids": NeuralType(('B', 'B', 'T'), ChannelType()),
            "p_input_mask": NeuralType(('B', 'B', 'T'), ChannelType()),
            "p_input_type_ids": NeuralType(('B', 'B', 'T'), ChannelType()),
        }

    def __init__(
        self,
        tokenizer,
        passages,
        queries,
        triples,
        batch_size,
        max_query_length=32,
        max_passage_length=192,
        num_negatives=1,
        dataset_type=BertDensePassageRetrievalDataset,
    ):
        dataset_params = {
            'tokenizer': tokenizer,
            'passages': passages,
            'queries': queries,
            'triples': triples,
            'max_query_length': max_query_length,
            'max_passage_length': max_passage_length,
            'num_negatives': num_negatives,
        }
        super().__init__(dataset_type, dataset_params, batch_size)


class BertDensePassageRetrievalDataLayerEval(TextDataLayer):

    @property
    @add_port_docs()
    def output_ports(self):
        return {
            "q_input_ids": NeuralType(('B', 'B', 'T'), ChannelType()),
            "q_input_mask": NeuralType(('B', 'B', 'T'), ChannelType()),
            "q_input_type_ids": NeuralType(('B', 'B', 'T'), ChannelType()),
            "p_input_ids": NeuralType(('B', 'B', 'T'), ChannelType()),
            "p_input_mask": NeuralType(('B', 'B', 'T'), ChannelType()),
            "p_input_type_ids": NeuralType(('B', 'B', 'T'), ChannelType()),
            "psg_rels": NeuralType(('B', 'T'), ChannelType()),
        }

    def __init__(
        self,
        tokenizer,
        passages,
        queries,
        qrels,
        topk_list,
        max_query_length=32,
        max_passage_length=192,
        num_candidates=10,
        dataset_type=BertDensePassageRetrievalDatasetEval,
    ):
        dataset_params = {
            'tokenizer': tokenizer,
            'passages': passages,
            'queries': queries,
            'qrels': qrels,
            'topk_list': topk_list,
            'max_query_length': max_query_length,
            'max_passage_length': max_passage_length,
            'num_candidates': num_candidates,
        }
        super().__init__(dataset_type, dataset_params, batch_size=1)
