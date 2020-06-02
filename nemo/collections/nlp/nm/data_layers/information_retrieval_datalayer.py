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


from nemo.collections.nlp.data import BertInformationRetrievalDatasetTrain, \
    BertInformationRetrievalDatasetEval, BertDensePassageRetrievalDatasetInfer
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import ChannelType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['BertInformationRetrievalDataLayerTrain',
           'BertInformationRetrievalDataLayerEval',
           'BertDensePassageRetrievalDataLayerTrain',
           'BertDensePassageRetrievalDataLayerEval',
           'BertDensePassageRetrievalDataLayerInfer']


class BertInformationRetrievalDataLayerTrain(TextDataLayer):
    """
    Data layer for training information retrieval system based on pre-trained Bert.

    Args:
        tokenizer (TokenizerSpec): Bert tokenizer
        passages (str): path to the collection of passages tsv file
        queries (str): path to the queries tsv file
        query_to_passages (str): path to the tsv file with training dataset
            with lines of the following form:
            [query_id] [rel_doc_id] [irrel_doc_1_id] ... [irrel_doc_n_id]
        batch_size (int): text segments batch size
        max_query_length (int): maximum allowed length of the query, default: 31
        max_passage_length (int): maximum allowed length of the passage, default: 190
        num_negatives (int): number of negative passages per query, default: 5
        shuffle (bool): whether to shuffle training data, default: True
        dataset_type (Dataset):
                the underlying dataset. Default: BertInformationRetrievalDatasetTrain
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids: tensor of shape [batch_size x num_docs x seq_len], each entry
            has form [CLS] query_tokens [SEP] passage_tokens [SEP]
        input_mask: bool tensor with 0s in place of pad tokens in input_ids to be masked
        input_type_ids: bool tensor with 0s for first query tokens and 1s for passage tokens
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
    Data layer for evaluating information retrieval system based on pre-trained Bert.

    Args:
        tokenizer (TokenizerSpec): Bert tokenizer
        passages (str): path to the collection of passages tsv file
        queries (str): path to the queries tsv file
        query_to_passages (str): path to the tsv file with training dataset
            with lines of the following form:
            [query_id] [doc_1_id] ... [doc_n_id]
        max_query_length (int): maximum allowed length of the query, default: 31
        max_passage_length (int): maximum allowed length of the passage, default: 190
        num_candidates (int): number of candidate passages per query in one batch, default: 10
        dataset_type (Dataset):
                the underlying dataset. Default: BertInformationRetrievalDatasetEval
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids: tensor of shape [batch_size x num_docs x seq_len], each entry
            has form [CLS] query_tokens [SEP] passage_tokens [SEP]
        input_mask: bool tensor with 0s in place of pad tokens in input_ids to be masked
        input_type_ids: bool tensor with 0s everywhere
        query_id: tensor of shape [batch_size, ] with ids of queries in a batch
        passage_ids: tensor of shape [batch_size, num_passages] with ids of passages in a batch
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
    Data layer for training dense passage retrieval system based on pre-trained Berts.

    Args:
        tokenizer (TokenizerSpec): Bert tokenizer
        passages (str): path to the collection of passages tsv file
        queries (str): path to the queries tsv file
        query_to_passages (str): path to the tsv file with training dataset
            with lines of the following form:
            [query_id] [rel_doc_id] [irrel_doc_1_id] ... [irrel_doc_n_id]
        batch_size (int): text segments batch size
        max_query_length (int): maximum allowed length of the query, default: 30
        max_passage_length (int): maximum allowed length of the passage, default: 190
        num_negatives (int): number of negative passages per query, default: 5
        shuffle (bool): whether to shuffle training data, default: True
        dataset_type (Dataset):
                the underlying dataset. Default: BertInformationRetrievalDatasetTrain
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
            'preprocess_fn': "preprocess_dpr",
        }
        super().__init__(dataset_type, dataset_params, batch_size, shuffle=shuffle)


class BertDensePassageRetrievalDataLayerEval(TextDataLayer):
    """
    Data layer for evaluating dense passage retrieval system based on pre-trained Berts.

    Args:
        tokenizer (TokenizerSpec): Bert tokenizer
        passages (str): path to the collection of passages tsv file
        queries (str): path to the queries tsv file
        query_to_passages (str): path to the tsv file with training dataset
            with lines of the following form:
            [query_id] [doc_1_id] ... [doc_n_id]
        max_query_length (int): maximum allowed length of the query, default: 31
        max_passage_length (int): maximum allowed length of the passage, default: 190
        num_candidates (int): number of candidate passages per query in one batch, default: 10
        dataset_type (Dataset):
                the underlying dataset. Default: BertInformationRetrievalDatasetEval
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
        query_id: tensor of shape [batch_size, ] with ids of queries in a batch
        passage_ids: tensor of shape [batch_size, num_passages] with ids of passages in a batch
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
    Data layer for inference with dense passage retrieval system based on pre-trained Berts.

    Args:
        tokenizer (TokenizerSpec): Bert tokenizer
        passages (str): path to the collection of passages tsv file
        queries (str): path to the queries tsv file
        batch_size (int): text segments batch size
        max_query_length (int): maximum allowed length of the query, default: 31
        max_passage_length (int): maximum allowed length of the passage, default: 190
        dataset_type (Dataset):
                the underlying dataset. Default: BertInformationRetrievalDatasetInfer
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids: tensor of shape [batch_size x 1 x seq_len], each entry
            has form of either [CLS] query_tokens [SEP] or [CLS] passage_tokens [SEP]
        input_mask: bool tensor with 0s in place of pad tokens in q_input_ids to be masked
        input_type_ids: tensor of 0s everywhere
        idx: tensor of shape [batch_size, ] with ids of either queries or passages
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), ChannelType()),
            "input_type_ids": NeuralType(('B', 'T'), ChannelType()),
            "idx": NeuralType(('B', ), ChannelType()),
        }

    def __init__(
        self,
        tokenizer,
        passages,
        queries,
        batch_size,
        max_query_length=30,
        max_passage_length=190,
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
