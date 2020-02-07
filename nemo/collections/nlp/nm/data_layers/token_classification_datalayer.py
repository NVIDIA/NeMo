# =============================================================================
# Copyright 2019 AI Applications Design Team at NVIDIA. All Rights Reserved.
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

from nemo.collections.nlp.data import BertTokenClassificationDataset, BertTokenClassificationInferDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import ChannelType, LabelsType, NeuralType

__all__ = ['BertTokenClassificationDataLayer', 'BertTokenClassificationInferDataLayer']


class BertTokenClassificationDataLayer(TextDataLayer):
    @property
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "loss_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "subtokens_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "labels": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_ids": NeuralType(ChannelType(), ('B', 'T')),
            "input_type_ids": NeuralType(ChannelType(), ('B', 'T')),
            "input_mask": NeuralType(ChannelType(), ('B', 'T')),
            "loss_mask": NeuralType(ChannelType(), ('B', 'T')),
            "subtokens_mask": NeuralType(ChannelType(), ('B', 'T')),
            "labels": NeuralType(LabelsType(), ('B', 'T')),
        }

    def __init__(
        self,
        text_file,
        label_file,
        tokenizer,
        max_seq_length,
        pad_label='O',
        label_ids=None,
        num_samples=-1,
        shuffle=False,
        batch_size=64,
        ignore_extra_tokens=False,
        ignore_start_end=False,
        use_cache=False,
        dataset_type=BertTokenClassificationDataset,
    ):
        dataset_params = {
            'text_file': text_file,
            'label_file': label_file,
            'max_seq_length': max_seq_length,
            'tokenizer': tokenizer,
            'num_samples': num_samples,
            'shuffle': shuffle,
            'pad_label': pad_label,
            'label_ids': label_ids,
            'ignore_extra_tokens': ignore_extra_tokens,
            'ignore_start_end': ignore_start_end,
            'use_cache': use_cache,
        }
        super().__init__(dataset_type, dataset_params, batch_size, shuffle)


class BertTokenClassificationInferDataLayer(TextDataLayer):
    @property
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "loss_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "subtokens_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_ids": NeuralType(ChannelType(), ('B', 'T')),
            "input_type_ids": NeuralType(ChannelType(), ('B', 'T')),
            "input_mask": NeuralType(ChannelType(), ('B', 'T')),
            "loss_mask": NeuralType(ChannelType(), ('B', 'T')),
            "subtokens_mask": NeuralType(ChannelType(), ('B', 'T')),
        }

    def __init__(
        self, queries, tokenizer, max_seq_length, batch_size=1, dataset_type=BertTokenClassificationInferDataset,
    ):
        dataset_params = {'queries': queries, 'tokenizer': tokenizer, 'max_seq_length': max_seq_length}
        super().__init__(dataset_type, dataset_params, batch_size, shuffle=False)
