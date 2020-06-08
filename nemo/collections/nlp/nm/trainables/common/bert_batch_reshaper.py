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


from nemo.backends.pytorch import TrainableNM
from nemo.core import ChannelType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['BertBatchReshaper']


class BertBatchReshaper(TrainableNM):
    @property
    @add_port_docs()
    def input_ports(self):
        return {
            "input_ids": NeuralType(('B', 'B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'B', 'T'), ChannelType()),
            "input_type_ids": NeuralType(('B', 'B', 'T'), ChannelType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), ChannelType()),
            "input_type_ids": NeuralType(('B', 'T'), ChannelType()),
        }

    def __init__(self):
        super().__init__()

    def forward(self, input_ids, input_mask, input_type_ids):
        seq_length = input_ids.shape[-1]
        input_ids = input_ids.view(-1, seq_length)
        input_mask = input_mask.view(-1, seq_length)
        input_type_ids = input_type_ids.view(-1, seq_length)
        return input_ids, input_mask, input_type_ids
