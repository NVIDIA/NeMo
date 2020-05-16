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
from nemo.collections.nlp.data import BertInformationRetrievalDatasetTrain, BertInformationRetrievalDatasetEval, \
    BertDensePassageRetrievalDatasetInfer
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import ChannelType, LabelsType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['BertInformationRetrievalDataLayerTrain',
           'BertInformationRetrievalDataLayerEval',
           'BertDensePassageRetrievalDataLayerTrain',
           'BertDensePassageRetrievalDataLayerEval',
           'BertDensePassageRetrievalDataLayerInfer']


class BertInformationRetrievalDataLayerTrain(TextDataLayer):
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

        input_ids: tensor of shape [batch_size x num_docs x seq_len], each entry
            has form [CLS] query_tokens [SEP] doc_tokens [SEP], first document in each
            batch corresponds to relevant document, all other to irrelevant ones
        input_mask: bool tensor with 0s in place of pad tokens in input_ids to be masked
        input_type_ids: 0 for query tokens and 1 for document tokens
        """
        return {
            "input_ids": NeuralType(('B', 'B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'B', 'T'), ChannelType()),
            "input_type_ids": NeuralType(('B', 'B', 'T'), ChannelType()),
        }

    def __init__(
        self,
        tokenizer,
        passages,
        queries,
        query_to_passages,
        batch_size,
        max_query_length=31,
        max_passage_length=190,
        num_negatives=5,
        shuffle=True,
        dataset_type=BertInformationRetrievalDatasetTrain,
    ):
        dataset_params = {
            'tokenizer': tokenizer,
            'passages': passages,
            'queries': queries,
            'query_to_passages': query_to_passages,
            'max_query_length': max_query_length,
            'max_passage_length': max_passage_length,
            'num_negatives': num_negatives,
        }
        super().__init__(dataset_type, dataset_params, batch_size, shuffle=shuffle)


class BertInformationRetrievalDataLayerEval(TextDataLayer):
    """
    Data layer for information retrieval with pre-trained Bert.

    Args:
        tokenizer (TokenizerSpec): Bert tokenizer
        documents (str): Path to the collection of documents tsv file
        queries (str): Path to the queries tsv file
        qrels (str): Path to the tsv file with query-document relevance information
        max_seq_length (int):
                maximum allowed length of the text segments. Default: 256
        dataset_type (Dataset):
                the underlying dataset. Default: BertInformationRetrievalDatasetEval
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids: tensor of shape [batch_size x num_docs x seq_len], each entry
            has form [CLS] query_tokens [SEP] doc_tokens [SEP]
        input_mask: bool tensor with 0s in place of pad tokens in input_ids to be masked
        input_type_ids: 0 for query tokens and 1 for document tokens
        doc_rels: tensor of shape [batch_size x num_docs] with 0 for irrelevant query-doc
            pairs and 1 for relevant pairs to compute MRR
        """
        return {
            "input_ids": NeuralType(('B', 'B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'B', 'T'), ChannelType()),
            "input_type_ids": NeuralType(('B', 'B', 'T'), ChannelType()),
            "query_id": NeuralType(('B', ), ChannelType()),
            "passage_ids": NeuralType(('B', 'T'), ChannelType()),
        }

    def __init__(
        self,
        tokenizer,
        passages,
        queries,
        query_to_passages,
        max_query_length=31,
        max_passage_length=190,
        num_candidates=10,
        dataset_type=BertInformationRetrievalDatasetEval,
    ):
        dataset_params = {
            'tokenizer': tokenizer,
            'passages': passages,
            'queries': queries,
            'query_to_passages': query_to_passages,
            'max_query_length': max_query_length,
            'max_passage_length': max_passage_length,
            'num_candidates': num_candidates,
        }
        super().__init__(dataset_type, dataset_params, batch_size=1)


class BertDensePassageRetrievalDataLayerTrain(TextDataLayer):
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

        q_input_ids: tensor of shape [batch_size x 1 x seq_len], each entry
            has form [CLS] query_tokens [SEP]
        q_input_mask: bool tensor with 0s in place of pad tokens in q_input_ids to be masked
        q_input_type_ids: tensor of 0s everywhere
        p_input_ids: tensor of shape [batch_size x num_psgs x seq_len], each entry
            has form [CLS] passage_tokens [SEP], first passage in each
            batch corresponds to relevant passage, all other to irrelevant ones
        p_input_mask: bool tensor with 0s in place of pad tokens in p_input_ids to be masked
        p_input_type_ids: tensor of 0s everywhere
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
        query_to_passages,
        batch_size,
        max_query_length=30,
        max_passage_length=190,
        num_negatives=1,
        shuffle=True,
        dataset_type=BertInformationRetrievalDatasetTrain,
    ):
        dataset_params = {
            'tokenizer': tokenizer,
            'passages': passages,
            'queries': queries,
            'query_to_passages': query_to_passages,
            'max_query_length': max_query_length,
            'max_passage_length': max_passage_length,
            'num_negatives': num_negatives,
            'preprocess_fn': "preprocess_dpr",
        }
        super().__init__(dataset_type, dataset_params, batch_size, shuffle=shuffle)


class BertDensePassageRetrievalDataLayerEval(TextDataLayer):

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        q_input_ids: tensor of shape [batch_size x 1 x seq_len], each entry
            has form [CLS] query_tokens [SEP]
        q_input_mask: bool tensor with 0s in place of pad tokens in q_input_ids to be masked
        q_input_type_ids: tensor of 0s everywhere
        p_input_ids: tensor of shape [batch_size x num_psgs x seq_len], each entry
            has form [CLS] passage_tokens [SEP], first passage in each
            batch corresponds to relevant passage, all other to irrelevant ones
        p_input_mask: bool tensor with 0s in place of pad tokens in p_input_ids to be masked
        p_input_type_ids: tensor of 0s everywhere
        doc_rels: tensor of shape [batch_size x num_psgs] with 0 for irrelevant query-passage
            pairs and 1 for relevant pairs to compute MRR
        """
        return {
            "q_input_ids": NeuralType(('B', 'B', 'T'), ChannelType()),
            "q_input_mask": NeuralType(('B', 'B', 'T'), ChannelType()),
            "q_input_type_ids": NeuralType(('B', 'B', 'T'), ChannelType()),
            "p_input_ids": NeuralType(('B', 'B', 'T'), ChannelType()),
            "p_input_mask": NeuralType(('B', 'B', 'T'), ChannelType()),
            "p_input_type_ids": NeuralType(('B', 'B', 'T'), ChannelType()),
            "query_id": NeuralType(('B', ), ChannelType()),
            "passage_ids": NeuralType(('B', 'T'), ChannelType()),
        }

    def __init__(
        self,
        tokenizer,
        passages,
        queries,
        query_to_passages,
        max_query_length=30,
        max_passage_length=190,
        num_candidates=10,
        dataset_type=BertInformationRetrievalDatasetEval,
    ):
        dataset_params = {
            'tokenizer': tokenizer,
            'passages': passages,
            'queries': queries,
            'query_to_passages': query_to_passages,
            'max_query_length': max_query_length,
            'max_passage_length': max_passage_length,
            'num_candidates': num_candidates,
            'preprocess_fn': "preprocess_dpr",
        }
        super().__init__(dataset_type, dataset_params, batch_size=1)


class BertDensePassageRetrievalDataLayerInfer(TextDataLayer):
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

        input_ids: tensor of shape [batch_size x 1 x seq_len], each entry
            has form [CLS] query_tokens [SEP]
        input_mask: bool tensor with 0s in place of pad tokens in q_input_ids to be masked
        input_type_ids: tensor of 0s everywhere
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), ChannelType()),
            "input_type_ids": NeuralType(('B', 'T'), ChannelType()),
            "idx": NeuralType(tuple('B'), ChannelType()),
        }

    def __init__(
        self,
        tokenizer,
        passages,
        queries,
        batch_size,
        max_query_length=32,
        max_passage_length=192,
        dataset_type=BertDensePassageRetrievalDatasetInfer,
    ):
        dataset_params = {
            'tokenizer': tokenizer,
            'passages': passages,
            'queries': queries,
            'max_query_length': max_query_length,
            'max_passage_length': max_passage_length,
        }
        super().__init__(dataset_type, dataset_params, batch_size)
