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

from nemo.collections.nlp.data import LanguageModelingDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import AxisType, BatchTag, NeuralType, TimeTag

__all__ = ['LanguageModelingDataLayer']


class LanguageModelingDataLayer(TextDataLayer):
    """
    Data layer for standard language modeling task.

    Args:
        dataset (str): path to text document with data
        tokenizer (TokenizerSpec): tokenizer
        max_seq_length (int): maximum allowed length of the text segments
        batch_step (int): how many tokens to skip between two successive
            segments of text when constructing batches
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids: indices of tokens which constitute batches of text segments
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_mask: bool tensor with 0s in place of tokens to be masked
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        labels: indices of tokens which should be predicted from each of the
            corresponding tokens in input_ids; for left-to-right language
            modeling equals to input_ids shifted by 1 to the right
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)
        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "labels": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
        }

    def __init__(
        self, dataset, tokenizer, max_seq_length, batch_size, batch_step=128, dataset_type=LanguageModelingDataset
    ):
        dataset_params = {
            'dataset': dataset,
            'tokenizer': tokenizer,
            'max_seq_length': max_seq_length,
            'batch_step': batch_step,
        }
        super().__init__(dataset_type, dataset_params, batch_size, shuffle=False)
