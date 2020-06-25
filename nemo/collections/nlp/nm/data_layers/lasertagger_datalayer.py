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

from nemo.collections.nlp.data.datasets.lasertagger_dataset import LaserTaggerDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import ChannelType, LabelsType, MaskType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['LaserTaggerDataLayer']


class LaserTaggerDataLayer(TextDataLayer):
    """
    Data layer for LaserTagger from source (src) to target (tgt) editing tasks.

    Args:
        preprocessed_data (str): path to preprocessed train/validation/test data
        use_t2t_decoder (bool): whether to use Autoregressive Decoder
        dataset_type (Dataset):
                the underlying dataset. Default: LaserTaggerDataset
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids: indices of tokens which constitute batches of masked text segments
        input_mask: bool tensor with 0s in place of source tokens to be masked
        segment_ids: bool tensor with 0's and 1's to denote the text segment type
        tgt_ids: indices of target tokens which constitute batches of masked text segments
        labels_mask: bool tensor with 0s in place of label tokens to be masked
        labels: indices of tokens which should be predicted from each of the
            corresponding target tokens in tgt_ids
        loss_mask: bool tensor with 0s in place of label tokens to be masked used for
            CrossEntropyLossNM

        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), ChannelType()),
            "segment_ids": NeuralType(('B', 'T'), ChannelType()),
            "tgt_ids": NeuralType(('B', 'T'), LabelsType()),
            "labels_mask": NeuralType(('B', 'T'), MaskType()),
            "labels": NeuralType(('B', 'T'), LabelsType()),
            "loss_mask": NeuralType(('B', 'T'), MaskType()),
        }

    def __init__(
        self, preprocessed_data, use_t2t_decoder, batch_size, shuffle=False, dataset_type=LaserTaggerDataset,
    ):
        dataset_params = {
            'preprocessed_data': preprocessed_data,
            'use_t2t_decoder': use_t2t_decoder,
        }
        super().__init__(dataset_type, dataset_params, batch_size=batch_size, shuffle=shuffle)
