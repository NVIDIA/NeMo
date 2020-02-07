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

from nemo.collections.nlp.data import BertTextClassificationDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import AxisType, BatchTag, NeuralType, TimeTag

__all__ = ['BertSentenceClassificationDataLayer']


class BertSentenceClassificationDataLayer(TextDataLayer):
    """
    Creates the data layer to use for the task of sentence classification
    with pretrained model.

    All the data processing is done BertSentenceClassificationDataset.

    Args:
        dataset (BertTextClassificationDataset):
                the dataset that needs to be converted to DataLayerNM
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_type_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        labels:
            0: AxisType(BatchTag)

        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "labels": NeuralType({0: AxisType(BatchTag)}),
        }

    def __init__(
        self,
        input_file,
        tokenizer,
        max_seq_length,
        num_samples=-1,
        shuffle=False,
        batch_size=64,
        dataset_type=BertTextClassificationDataset,
    ):
        dataset_params = {
            'input_file': input_file,
            'tokenizer': tokenizer,
            'max_seq_length': max_seq_length,
            'num_samples': num_samples,
            'shuffle': shuffle,
        }
        super().__init__(dataset_type, dataset_params, batch_size, shuffle)
