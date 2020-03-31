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

from nemo.collections.nlp.data import GLUEDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import CategoricalValuesType, ChannelType, NeuralType, RegressionValuesType
from nemo.utils.decorators import add_port_docs

__all__ = ['GlueClassificationDataLayer', 'GlueRegressionDataLayer']


class GlueClassificationDataLayer(TextDataLayer):
    """
    Creates the data layer to use for the GLUE classification tasks,
    more details here: https://gluebenchmark.com/tasks

    All the data processing is done in GLUEDataset.

    Args:
        data_dir (str): data directory path
        tokenizer (TokenizerSpec): text tokenizer.
        max_seq_length (int): maximum allowed length of the text segments .
        processor (DataProcessor): data processor.
        evaluate (bool): true if data layer is used for evaluation. Default: False.
        batch_size (int): batch size in segments
        shuffle (bool): whether to shuffle data or not. Default: False.
        dataset_type (GLUEDataset):
                the dataset that needs to be converted to DataLayerNM
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids:
            indices of tokens which constitute batches of text segments
        input_type_ids:
            tensor with 0's and 1's to denote the text segment type
        input_mask:
            bool tensor with 0s in place of tokens to be masked
        labels:
            integer indices for sentence classication prediction
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_type_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), ChannelType()),
            "labels": NeuralType(tuple('B'), CategoricalValuesType()),
        }

    def __init__(
        self,
        data_dir,
        tokenizer,
        max_seq_length,
        processor,
        evaluate=False,
        shuffle=False,
        batch_size=64,
        dataset_type=GLUEDataset,
        use_data_cache=False,
    ):
        dataset_params = {
            'data_dir': data_dir,
            'output_mode': 'classification',
            'processor': processor,
            'evaluate': evaluate,
            'tokenizer': tokenizer,
            'max_seq_length': max_seq_length,
            'use_data_cache': use_data_cache,
        }
        super().__init__(dataset_type, dataset_params, batch_size, shuffle)


class GlueRegressionDataLayer(TextDataLayer):
    """
    Creates the data layer to use for the GLUE STS-B regression task,
    more details here: https://gluebenchmark.com/tasks

    All the data processing is done in GLUEDataset.

    Args:
        data_dir (str): data directory path
        tokenizer (TokenizerSpec): text tokenizer.
        max_seq_length (int): maximum allowed length of the text segments .
        processor (DataProcessor): data processor.
        evaluate (bool): true if data layer is used for evaluation. Default: False.
        batch_size (int): batch size in segments
        shuffle (bool): whether to shuffle data or not. Default: False.
        dataset_type (GLUEDataset):
                the dataset that needs to be converted to DataLayerNM
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids:
            indices of tokens which constitute batches of text segments
        input_type_ids:
            tensor with 0's and 1's to denote the text segment type
        input_mask:
            bool tensor with 0s in place of tokens to be masked
        labels:
            float for sentence regression prediction
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_type_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), ChannelType()),
            "labels": NeuralType(tuple('B'), RegressionValuesType()),
        }

    def __init__(
        self,
        data_dir,
        tokenizer,
        max_seq_length,
        processor,
        evaluate=False,
        shuffle=False,
        batch_size=64,
        dataset_type=GLUEDataset,
        use_data_cache=False,
    ):
        dataset_params = {
            'data_dir': data_dir,
            'output_mode': 'regression',
            'processor': processor,
            'evaluate': evaluate,
            'tokenizer': tokenizer,
            'max_seq_length': max_seq_length,
            'use_data_cache': use_data_cache,
        }

        super().__init__(dataset_type, dataset_params, batch_size, shuffle)
