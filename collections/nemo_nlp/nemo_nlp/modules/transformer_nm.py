# Copyright (c) 2019 NVIDIA Corporation
"""
This package contains Transformer for translation Neural Module
"""
__all__ = ['TransformerEncoderNM',
           'TransformerDecoderNM',
           'GreedyLanguageGeneratorNM',
           'BeamSearchTranslatorNM']

import math

from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_types import *

from ..transformer import (TransformerEmbedding,
                           TransformerEncoder,
                           TransformerDecoder,
                           GreedySequenceGenerator,
                           BeamSearchSequenceGenerator)
from ..transformer.utils import transformer_weights_init


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

    @staticmethod
    def create_ports():
        input_ports = {
            "input_ids": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_mask_src": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
        }

        output_ports = {
            "hidden_states": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            })
        }
        return input_ports, output_ports

    def __init__(self,
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
                 **kwargs):
        TrainableNM.__init__(self, **kwargs)

        self.embedding_layer = TransformerEmbedding(
            vocab_size=vocab_size,
            hidden_size=d_model,
            max_sequence_length=max_seq_length,
            embedding_dropout=embedding_dropout,
            learn_positional_encodings=learn_positional_encodings)
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            hidden_size=d_model,
            mask_future=mask_future,
            num_attention_heads=num_attn_heads,
            inner_size=d_inner,
            ffn_dropout=ffn_dropout,
            hidden_act=hidden_act,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout)

        std_init_range = 1 / math.sqrt(d_model)
        self.apply(lambda module: transformer_weights_init(module,
                                                           std_init_range))
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

    @staticmethod
    def create_ports():
        input_ports = {
            "input_ids_tgt": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "hidden_states_src": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
            "input_mask_src": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_mask_tgt": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
        }

        output_ports = {
            "hidden_states": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            })
        }
        return input_ports, output_ports

    def __init__(self,
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
                 **kwargs):
        TrainableNM.__init__(self, **kwargs)

        self.embedding_layer = TransformerEmbedding(
            vocab_size=vocab_size,
            hidden_size=d_model,
            max_sequence_length=max_seq_length,
            embedding_dropout=embedding_dropout,
            learn_positional_encodings=learn_positional_encodings
        )
        self.decoder = TransformerDecoder(
            num_layers=num_layers,
            hidden_size=d_model,
            num_attention_heads=num_attn_heads,
            inner_size=d_inner,
            ffn_dropout=ffn_dropout,
            hidden_act=hidden_act,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout
        )

        std_init_range = 1 / math.sqrt(d_model)
        self.apply(lambda module: transformer_weights_init(module,
                                                           std_init_range))
        self.to(self._device)

    def forward(self,
                input_ids_tgt,
                hidden_states_src,
                input_mask_src,
                input_mask_tgt):
        hidden_states_tgt = self.embedding_layer(input_ids_tgt)
        hidden_states = self.decoder(hidden_states_tgt,
                                     input_mask_tgt,
                                     hidden_states_src,
                                     input_mask_src)
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

    @staticmethod
    def create_ports():
        input_ports = {
            "input_ids": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            })
        }

        output_ports = {
            "output_ids": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            })
        }
        return input_ports, output_ports

    def __init__(self,
                 decoder,
                 log_softmax,
                 max_seq_length,
                 pad_token,
                 bos_token,
                 eos_token,
                 batch_size=1,
                 **kwargs):
        TrainableNM.__init__(self, **kwargs)

        self.generator = GreedySequenceGenerator(
            decoder.embedding_layer,
            decoder.decoder,
            log_softmax,
            max_sequence_length=max_seq_length,
            pad=pad_token,
            bos=bos_token,
            eos=eos_token,
            batch_size=batch_size
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

    @staticmethod
    def create_ports():
        input_ports = {
            "hidden_states_src": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
            "input_mask_src": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            })
        }

        output_ports = {
            "output_ids": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            })
        }
        return input_ports, output_ports

    @property
    def num_weights(self):
        return 0

    def __init__(self,
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
                 **kwargs):
        TrainableNM.__init__(self, **kwargs)

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
            len_pen=length_penalty
        )

    def forward(self, hidden_states_src, input_mask_src):
        output_ids = self.generator(encoder_hidden_states=hidden_states_src,
                                    encoder_input_mask=input_mask_src)
        return output_ids
