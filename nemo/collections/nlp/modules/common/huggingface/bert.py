# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict, Optional

from transformers import BertModel

from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import ChannelType, NeuralType
from nemo.utils.decorators import experimental

__all__ = ['BertEncoder']


@experimental
class BertModule(NeuralModule):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "token_type_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('B', 'T'), ChannelType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "last_hidden_states": NeuralType(('B', 'T', 'D'), ChannelType()),
            "pooler_output ": NeuralType(('B', 'D'), ChannelType()),
            "hidden_states ": NeuralType(('B', 'T', 'D'), ChannelType()),
            "attentions ": NeuralType(('B', 'H', 'T', 'D'), ChannelType()),
        }


@experimental
class BertEncoder(BertModel, BertModule):
    """
    ALBERT wraps around the Huggingface implementation of ALBERT from their
    transformers repository for easy use within NeMo.
    Args:
        pretrained_model_name (str): If using a pretrained model, this should
            be the model's name. Otherwise, should be left as None.
        config_filename (str): path to model configuration file. Optional.
        vocab_size (int): Size of the vocabulary file, if not using a
            pretrained model.
        hidden_size (int): Size of the encoder and pooler layers.
        num_hidden_layers (int): Number of hidden layers in the encoder.
        num_attention_heads (int): Number of attention heads for each layer.
        intermediate_size (int): Size of intermediate layers in the encoder.
        hidden_act (str): Activation function for encoder and pooler layers;
            "gelu", "relu", and "swish" are supported.
        max_position_embeddings (int): The maximum number of tokens in a
        sequence.
    """

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    @typecheck()
    def forward(self, **kwargs):
        res = super().forward(**kwargs)
        return res
