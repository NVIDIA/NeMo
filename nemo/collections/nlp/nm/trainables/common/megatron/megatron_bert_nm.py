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

# from megatron.model.language_model import get_language_model
# from megatron.model.bert_model import bert_attention_mask_func, bert_extended_attention_mask, bert_position_ids
# from megatron.model.utils import init_method_normal, scaled_init_method_normal

from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_types import ChannelType, NeuralType
from nemo.utils.decorators import add_port_docs

import argparse
import json
import os

import sys
sys.path.append('/home/ebakhturina/megatron-lm')
from megatron.model import language_model
from megatron.model.language_model import get_language_model
from megatron.model.bert_model import bert_attention_mask_func, bert_extended_attention_mask, bert_position_ids
from megatron.model.utils import init_method_normal, scaled_init_method_normal
from megatron.model.bert_model import bert_attention_mask_func, bert_extended_attention_mask, bert_position_ids
from megatron.model.language_model import get_language_model
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal, scaled_init_method_normal

from megatron import get_args


__all__ = ['MegatronBERT']


class MegatronBERT(TrainableNM):
    """
    BERT wraps around the Huggingface implementation of BERT from their
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
            "attention_mask": NeuralType(('B', 'T'), ChannelType()),
            "token_type_ids": NeuralType(('B', 'T'), ChannelType(), optional=True),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        hidden_states: output embedding 
        """
        return {"hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}

    def __init__(
         self,
         config,
         init_method_std=0.02,
         num_tokentypes=2):

        super().__init__()

        import sys
        sys.path.append('/home/ebakhturina/megatron-lm')
        from megatron import get_args
        from megatron.initialize import initialize_megatron
        
        if not os.path.exists(config):
            raise ValueError (f'Config file not found at {config}')
        with open(config) as json_file:
            config = json.load(json_file)
        megatron_args = {"num_layers": config['num-layers'],
                         "hidden_size": config['hidden-size'],
                         "num_attention_heads": config['num-attention-heads'], 
                         "max_position_embeddings": config['max-seq-length'],
                         "padded_vocab_size": config['vocab-size']}
        
        initialize_megatron(None, megatron_args)
        init_method = init_method_normal(init_method_std)
          
        self.language_model, _ = get_language_model(
            attention_mask_func=bert_attention_mask_func,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            init_method=init_method,
            scaled_init_method=scaled_init_method_normal(init_method_std,
                                                         config['num-layers']))

        self.language_model.to(self._device)
        self._hidden_size = self.language_model.hidden_size

    @property
    def hidden_size(self):
        """
            Property returning hidden size.

            Returns:
                Hidden size.
        """
        return self._hidden_size


    def forward(self, input_ids, attention_mask, token_type_ids):
        import sys
        sys.path.append('/home/ebakhturina/scripts')
        from mem_report import mem_report
        import nemo
        nemo.logging.info(f'INSIDE before forward: {mem_report()}')
        extended_attention_mask = bert_extended_attention_mask(
            attention_mask, next(self.language_model.parameters()).dtype)
        position_ids = bert_position_ids(input_ids)
        nemo.logging.info(f'INSIDE after ext_att_mask: {mem_report()}')
        
        sequence_output = self.language_model(input_ids,
                                              position_ids,
                                              extended_attention_mask,
                                              tokentype_ids=token_type_ids)

        nemo.logging.info(f'INSIDE after forward: {mem_report()}')
        return sequence_output
