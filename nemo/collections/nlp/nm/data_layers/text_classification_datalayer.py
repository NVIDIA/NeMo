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

from nemo.collections.nlp.data.datasets.text_classification.text_classification_dataset import (
    BertTextClassificationDataset,
)
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import ChannelType, LabelsType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['BertTextClassificationDataLayer']


class BertTextClassificationDataLayer(TextDataLayer):
    """
    Creates the data layer to use for the task of sentence classification
    with pretrained model.

    All the data processing is done BertTextClassificationDataset.

    Args:
        input_file (str): data file
        tokenizer (TokenizerSpec): text tokenizer.
        max_seq_length (int): max sequence length minus 2 for [CLS] and [SEP]
        num_samples:
            TODO
        shuffle (bool): whether to shuffle data or not. Default: False.
        batch_size: text segments batch size
        dataset (BertTextClassificationDataset):
                the dataset that needs to be converted to DataLayerNM
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        input_ids:
            indices of tokens which constitute batches of masked text segments
        input_type_ids:
            tensor with 0's and 1's to denote the text segment type
        input_mask:
            bool tensor with 0s in place of tokens to be masked
        labels: sequence classification id
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_type_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), ChannelType()),
            "labels": NeuralType(tuple('B'), LabelsType()),
        }

    def __init__(
        self,
        input_file,
        tokenizer,
        max_seq_length,
        num_samples=-1,
        shuffle=False,
        batch_size=64,
        use_cache=False,
        dataset_type=BertTextClassificationDataset,
    ):
        dataset_params = {
            'input_file': input_file,
            'tokenizer': tokenizer,
            'max_seq_length': max_seq_length,
            'num_samples': num_samples,
            'use_cache': use_cache,
        }
        super().__init__(dataset_type, dataset_params, batch_size, shuffle=shuffle)
