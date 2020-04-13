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

from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import ChannelType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['BertInferDataLayer']


class BertInferDataLayer(TextDataLayer):
    """
    Data layer to run infernce with BERT (get final hidden layer).

    Args:
        tokenizer (TokenizerSpec): tokenizer
        dataset (str): directory or a single file with dataset documents
        max_seq_length (int): maximum allowed length of the text segments
        mask_probability (float): probability of masking input sequence tokens
        batch_size (int): batch size in segments
        short_seeq_prob (float): Probability of creating sequences which are
            shorter than the maximum length.
            Defualts to 0.1.
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids: indices of tokens which constitute batches of text segments
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_type_ids: indices of token types (e.g., sentences A & B in BERT)
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_mask: bool tensor with 0s in place of tokens to be masked
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_type_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), ChannelType()),
        }

    def __init__(self, dataset_type, dataset_params, batch_size=1, shuffle=False):

        super().__init__(dataset_type, dataset_params, batch_size=batch_size, shuffle=shuffle)
