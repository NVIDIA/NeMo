# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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


import math

from nemo.backends.pytorch.nm import TrainableNM
from nemo.collections.nlp.nm.trainables.common.transformer.transformer_decoders import TransformerDecoder
from nemo.collections.nlp.nm.trainables.common.transformer.transformer_encoders import TransformerEncoder
from nemo.collections.nlp.nm.trainables.common.transformer.transformer_generators import (
    BeamSearchSequenceGenerator,
    GreedySequenceGenerator,
)
from nemo.collections.nlp.nm.trainables.common.transformer.transformer_modules import TransformerEmbedding
from nemo.collections.nlp.utils.transformer_utils import transformer_weights_init
from nemo.core.neural_types import ChannelType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['TransformerEncoderNM', 'TransformerDecoderNM', 'GreedyLanguageGeneratorNM', 'BeamSearchTranslatorNM']


class TransformerEncoderNM(TrainableNM):
    """
    Neural module which consists of embedding layer followed by Transformer
    encoder.

    Args:
        vocab_size: size of the vocabulary (number of tokens)
        d_model: hidden size (d_model) of the Transformer
        max_seq_length: maximum allowed length of input sequences, feeding
            longer sequences will cause an error
        embedding_dropout: dropout ratio applied to embeddings
        learn_positional_encodings: bool, whether to learn positional encoding
            or use fixed sinusoidal encodings
        num_layers: number of layers in Transformer encoder
        mask_future: bool, whether to apply triangular future masking to the
            sequence of hidden states (which allows to use it for LM)
        num_attn_heads: number of attention heads
        d_inner: number of neurons in the intermediate part of feed-forward
            network (FFN)
        ffn_dropout: dropout ratio applied to FFN
        attn_score_dropout: dropout ratio applied to attention scores
        attn_layer_dropout: dropout ratio applied to the output of attn layer
        hidden_act: activation function applied in intermediate FFN module
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        input_ids: ids of input tokens
        input_mask_src: input mask
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask_src": NeuralType(('B', 'T'), ChannelType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        hidden_states: outputs hidden states
        """
        return {"hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}

    def __init__(
        self,
        vocab_size,
        d_model,
        d_inner,
        max_seq_length,
        num_layers,
        num_attn_heads,
        ffn_dropout=0.0,
        embedding_dropout=0.0,
        attn_score_dropout=0.0,
        attn_layer_dropout=0.0,
        learn_positional_encodings=False,
        hidden_act='relu',
        mask_future=False,
    ):
        super().__init__()

        self.embedding_layer = TransformerEmbedding(
            vocab_size=vocab_size,
            hidden_size=d_model,
            max_sequence_length=max_seq_length,
            embedding_dropout=embedding_dropout,
            learn_positional_encodings=learn_positional_encodings,
        )
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            hidden_size=d_model,
            mask_future=mask_future,
            num_attention_heads=num_attn_heads,
            inner_size=d_inner,
            ffn_dropout=ffn_dropout,
            hidden_act=hidden_act,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
        )

        std_init_range = 1 / math.sqrt(d_model)
        self.apply(lambda module: transformer_weights_init(module, std_init_range))
        self.to(self._device)

    def forward(self, input_ids, input_mask_src):
        hidden_states = self.embedding_layer(input_ids)
        hidden_states = self.encoder(hidden_states, input_mask_src)
        return hidden_states


class TransformerDecoderNM(TrainableNM):
    """
    Neural module which consists of embedding layer followed by Transformer
    decoder.

    Args:
        vocab_size: size of the vocabulary (number of tokens)
        d_model: hidden size (d_model) of the Transformer
        max_seq_length: maximum allowed length of input sequences, feeding
            longer sequences will cause an error
        embedding_dropout: dropout ratio applied to embeddings
        learn_positional_encodings: bool, whether to learn positional encoding
            or use fixed sinusoidal encodings
        num_layers: number of layers in Transformer decoder
        num_attn_heads: number of attention heads
        d_inner: number of neurons in the intermediate part of feed-forward
            network (FFN)
        ffn_dropout: dropout ratio applied to FFN
        attn_score_dropout: dropout ratio applied to attention scores
        attn_layer_dropout: dropout ratio applied to the output of attn layer
        hidden_act: activation function applied in intermediate FFN module
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        input_ids_tgt: ids of target sequence
        hidden_states_src: input hidden states 
        input_mask_src: input token mask
        input_mask_tgt: target token mask
        """
        return {
            "input_ids_tgt": NeuralType(('B', 'T'), ChannelType()),
            "hidden_states_src": NeuralType(('B', 'T', 'D'), ChannelType()),
            "input_mask_src": NeuralType(('B', 'T'), ChannelType()),
            "input_mask_tgt": NeuralType(('B', 'T'), ChannelType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        hidden_states: output hidden states
        """
        return {"hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}

    def __init__(
        self,
        vocab_size,
        d_model,
        d_inner,
        num_layers,
        max_seq_length,
        num_attn_heads,
        ffn_dropout=0.0,
        embedding_dropout=0.0,
        attn_score_dropout=0.0,
        attn_layer_dropout=0.0,
        learn_positional_encodings=False,
        hidden_act='relu',
    ):
        super().__init__()

        self.embedding_layer = TransformerEmbedding(
            vocab_size=vocab_size,
            hidden_size=d_model,
            max_sequence_length=max_seq_length,
            embedding_dropout=embedding_dropout,
            learn_positional_encodings=learn_positional_encodings,
        )
        self.decoder = TransformerDecoder(
            num_layers=num_layers,
            hidden_size=d_model,
            num_attention_heads=num_attn_heads,
            inner_size=d_inner,
            ffn_dropout=ffn_dropout,
            hidden_act=hidden_act,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
        )

        std_init_range = 1 / math.sqrt(d_model)
        self.apply(lambda module: transformer_weights_init(module, std_init_range))
        self.to(self._device)

    def forward(self, input_ids_tgt, hidden_states_src, input_mask_src, input_mask_tgt):
        hidden_states_tgt = self.embedding_layer(input_ids_tgt)
        hidden_states = self.decoder(hidden_states_tgt, input_mask_tgt, hidden_states_src, input_mask_src)
        return hidden_states


class GreedyLanguageGeneratorNM(TrainableNM):
    """
    Neural module for greedy text generation with language model

    Args:
        decoder: module which maps input_ids into hidden_states
        log_softmax: module which maps hidden_states into log_probs
        max_seq_length: maximum allowed length of generated sequences
        pad_token: index of padding token in the vocabulary
        bos_token: index of beginning of sequence token in the vocabulary
        eos_token: index of end of sequence token in the vocabulary
        batch_size: size of the batch of generated sequences if no starting
            tokens are provided
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        input_ids:  input ids
        """
        return {"input_ids": NeuralType(('B', 'T'), ChannelType())}

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        output ids: output ids
        """
        return {"output_ids": NeuralType(('B', 'T'), ChannelType())}

    def __init__(self, decoder, log_softmax, max_seq_length, pad_token, bos_token, eos_token, batch_size=1):
        super().__init__()

        self.generator = GreedySequenceGenerator(
            decoder.embedding_layer,
            decoder.decoder,
            log_softmax,
            max_sequence_length=max_seq_length,
            pad=pad_token,
            bos=bos_token,
            eos=eos_token,
            batch_size=batch_size,
        )

    @property
    def num_weights(self):
        return 0

    def forward(self, input_ids):
        output_ids = self.generator(decoder_input_ids=input_ids)
        return output_ids


class BeamSearchTranslatorNM(TrainableNM):
    """
    Neural module for beam search translation generation

    Args:
        decoder: module which maps input_ids into hidden_states
        log_softmax: module which maps hidden_states into log_probs
        max_seq_length: maximum allowed length of generated sequences
        pad_token: index of padding token in the vocabulary
        bos_token: index of beginning of sequence token in the vocabulary
        eos_token: index of end of sequence token in the vocabulary
        batch_size: size of the batch of generated sequences if no starting
            tokens are provided
        beam_size: size of the beam
        max_delta_length: maximum allowed difference between generated output
            and input sequence in case of conditional decoding
        length_penalty: parameter which penalizes shorter sequences
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        hidden_states_src: input hidden states
        input_mask_src: input mask
        """
        return {
            "hidden_states_src": NeuralType(('B', 'T', 'C'), ChannelType()),
            "input_mask_src": NeuralType(('B', 'T'), ChannelType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        output_ids: output ids
        """
        return {"output_ids": NeuralType(('B', 'T'), ChannelType())}

    @property
    def num_weights(self):
        return 0

    def __init__(
        self,
        decoder,
        log_softmax,
        max_seq_length,
        pad_token,
        bos_token,
        eos_token,
        batch_size=1,
        beam_size=4,
        max_delta_length=50,
        length_penalty=0,
    ):
        super().__init__()

        self.generator = BeamSearchSequenceGenerator(
            decoder.embedding_layer,
            decoder.decoder,
            log_softmax,
            max_sequence_length=max_seq_length,
            max_delta_length=max_delta_length,
            pad=pad_token,
            bos=bos_token,
            eos=eos_token,
            batch_size=batch_size,
            beam_size=beam_size,
            len_pen=length_penalty,
        )

    def forward(self, hidden_states_src, input_mask_src):
        output_ids = self.generator(encoder_hidden_states=hidden_states_src, encoder_input_mask=input_mask_src)
        return output_ids
