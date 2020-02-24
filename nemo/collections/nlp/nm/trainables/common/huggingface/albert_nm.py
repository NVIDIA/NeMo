# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
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

from typing import List, Optional

from transformers import (
    ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    AlbertConfig,
    AlbertModel,
)

from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_modules import PretrainedModelInfo
from nemo.core.neural_types import ChannelType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['Albert']


class Albert(TrainableNM):
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

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        input_ids: input token ids
        token_type_ids: segment type ids
        attention_mask: attention mask
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "token_type_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('B', 'T'), ChannelType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module input ports.
        hidden_states: output embedding
        """
        return {"hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}

    def __init__(
        self,
        pretrained_model_name=None,
        config_filename=None,
        vocab_size=None,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        max_position_embeddings=512,
    ):
        super().__init__()

        # Check that only one of pretrained_model_name, config_filename, and
        # vocab_size was passed in
        total = 0
        if pretrained_model_name is not None:
            total += 1
        if config_filename is not None:
            total += 1
        if vocab_size is not None:
            total += 1

        if total != 1:
            raise ValueError(
                "Only one of pretrained_model_name, vocab_size, "
                + "or config_filename should be passed into the "
                + "ALBERT constructor."
            )

        # TK: The following code checks the same once again.
        if vocab_size is not None:
            config = AlbertConfig(
                vocab_size_or_config_json_file=vocab_size,
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                max_position_embeddings=max_position_embeddings,
            )
            model = AlbertModel(config)
        elif pretrained_model_name is not None:
            model = AlbertModel.from_pretrained(pretrained_model_name)
        elif config_filename is not None:
            config = AlbertConfig.from_json_file(config_filename)
            model = AlbertModel(config)
        else:
            raise ValueError(
                "Either pretrained_model_name or vocab_size must" + " be passed into the ALBERT constructor"
            )

        model.to(self._device)

        self.add_module("albert", model)
        self.config = model.config
        self._hidden_size = model.config.hidden_size

    @property
    def hidden_size(self):
        """
            Property returning hidden size.
            Returns:
                Hidden size.
        """
        return self._hidden_size

    @staticmethod
    def list_pretrained_models() -> Optional[List[PretrainedModelInfo]]:
        pretrained_models = []
        for key, value in ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP.items():
            model_info = PretrainedModelInfo(
                pretrained_model_name=key,
                description="weights by HuggingFace",
                parameters=ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP[key],
                location=value,
            )
            pretrained_models.append(model_info)
        return pretrained_models

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.albert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
