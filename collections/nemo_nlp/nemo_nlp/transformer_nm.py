# Copyright (c) 2019 NVIDIA Corporation
"""
This package contains Transformer for translation Neural Module
"""
import math
from nemo.backends.pytorch.nm import TrainableNM, LossNM
from nemo.core.neural_types import *
from .transformer import TransformerEmbedding, TransformerEncoder, \
    TransformerDecoder, TransformerLogSoftmax, SmoothedCrossEntropyLoss, \
    GreedySequenceGenerator, BeamSearchSequenceGenerator
from .transformer.utils import mask_padded_tokens, transformer_weights_init


class TransformerEncoderNM(TrainableNM):
    """
    Neural module which consists of embedding layer followed by Transformer
    encoder.

    Args:
        vocab_size: size of the vocabulary (number of tokens)
        hidden_size: hidden size (d_model) of the Transformer
        max_sequence_length: maximum allowed length of input sequences, feeding
            longer sequences will cause an error
        embedding_dropout: dropout ratio applied to embeddings
        learn_positional_encodings: bool, whether to learn positional encoding
            or use fixed sinusoidal encodings
        num_layers: number of layers in Transformer encoder
        mask_future: bool, whether to apply triangular future masking to the
            sequence of hidden states (which allows to use it for LM)
        first_sub_layer: type of the first sublayer, surrently only
            self_attention and lightweight_conv are supported
        num_attention_heads: number of attention heads
        inner_size: number of neurons in the intermediate part of
            fully-connected network (second_sub_layer)
        ffn_dropout: dropout ratio applied to FFN
        attn_score_dropout: dropout ratio applied to attention scores
        attn_layer_dropout: dropout ratio applied to the output of attn layer
        conv_kernel_size: convolution kernel size in lightweight_conv
        conv_weight_dropout: dropout ratio applied to the convolution kernel
        conv_layer_dropout: dropout ratio applied to the output of conv layer
    """

    @staticmethod
    def create_ports():
        input_ports = {
            "input_ids":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_mask_src":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
        }

        output_ports = {
            "hidden_states":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            })
        }
        return input_ports, output_ports

    def __init__(self, **kwargs):
        TrainableNM.__init__(self, **kwargs)

        params = self.local_parameters
        embedding_params = {
            "vocab_size": params["vocab_size"],
            "hidden_size": params["d_model"],
            "max_sequence_length": params["max_seq_length"],
            "embedding_dropout": params.get("embedding_dropout", 0),
            "learn_positional_encodings":
                params.get("learn_positional_encodings", False)
        }
        backbone_params = {
            "num_layers": params["num_layers"],
            "hidden_size": params["d_model"],
            "mask_future": params.get("mask_future", False),
            "num_attention_heads": params["num_attn_heads"],
            "inner_size": params["d_inner"],
            "ffn_dropout": params.get("ffn_dropout", 0),
            "hidden_act": params.get("hidden_act", "relu"),
            "attn_score_dropout": params.get("attn_score_dropout", 0),
            "attn_layer_dropout": params.get("attn_layer_dropout", 0)
        }

        self.embedding_layer = TransformerEmbedding(**embedding_params)
        self.encoder = TransformerEncoder(**backbone_params)

        std_init_range = 1 / math.sqrt(params["d_model"])
        self.apply(
            lambda module: transformer_weights_init(module, std_init_range))
        self.to(self._device)

    def forward(self, input_ids, input_mask_src):
        hidden_states = self.embedding_layer(input_ids)
        hidden_states = self.encoder(hidden_states, input_mask_src)
        return hidden_states


class TransformerDecoderNM(TrainableNM):
    @staticmethod
    def create_ports():
        input_ports = {
            "input_ids_tgt":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "hidden_states_src":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
            "input_mask_src":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_mask_tgt":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
        }

        output_ports = {
            "hidden_states":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            })
        }
        return input_ports, output_ports

    def __init__(self, **kwargs):
        TrainableNM.__init__(self, **kwargs)

        params = self.local_parameters
        embedding_params = {
            "vocab_size": params["vocab_size"],
            "hidden_size": params["d_model"],
            "max_sequence_length": params["max_seq_length"],
            "embedding_dropout": params.get("embedding_dropout", 0),
            "learn_positional_encodings":
                params.get("learn_positional_encodings", False)
        }
        backbone_params = {
            "num_layers": params["num_layers"],
            "hidden_size": params["d_model"],
            "num_attention_heads": params["num_attn_heads"],
            "inner_size": params["d_inner"],
            "ffn_dropout": params.get("ffn_dropout", 0),
            "hidden_act": params.get("hidden_act", "relu"),
            "attn_score_dropout": params.get("attn_score_dropout", 0),
            "attn_layer_dropout": params.get("attn_layer_dropout", 0)
        }

        self.embedding_layer = TransformerEmbedding(**embedding_params)
        self.decoder = TransformerDecoder(**backbone_params)

        std_init_range = 1 / math.sqrt(params["d_model"])
        self.apply(
            lambda module: transformer_weights_init(module, std_init_range))
        self.to(self._device)

    def forward(self, input_ids_tgt, hidden_states_src, input_mask_src,
                input_mask_tgt):
        hidden_states_tgt = self.embedding_layer(input_ids_tgt)
        hidden_states = self.decoder(
            hidden_states_tgt, input_mask_tgt,
            hidden_states_src, input_mask_src)
        return hidden_states


class TransformerLogSoftmaxNM(TrainableNM):
    @staticmethod
    def create_ports():
        input_ports = {
            "hidden_states":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
        }

        output_ports = {
            "log_probs":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
        }
        return input_ports, output_ports

    def __init__(self, *, vocab_size, d_model, **kwargs):
        TrainableNM.__init__(self, **kwargs)

        self.log_softmax = TransformerLogSoftmax(
            vocab_size=vocab_size,
            hidden_size=d_model)

        self.log_softmax.apply(transformer_weights_init)
        self.log_softmax.to(self._device)

    def forward(self, hidden_states):
        log_probs = self.log_softmax(hidden_states)
        return log_probs


class GreedyLanguageGeneratorNM(TrainableNM):
    """
    Neural module for greedy text generation with language model

    Args:
        decoder: module which maps input_ids into hidden_states
        log_softmax: module which maps hidden_states into log_probs
        max_sequence_length: maximum allowed length of generated sequences
        pad: index of padding token in the vocabulary
        bos: index of beginning of sequence token in the vocabulary
        eos: index of end of sequence token in the vocabulary
        device: torch.device to conduct generation on
        batch_size: size of the batch of generated sequences if no starting
            tokens are provided
    """

    @staticmethod
    def create_ports():
        input_ports = {
            "input_ids":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            })
        }

        output_ports = {
            "output_ids":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            })
        }
        return input_ports, output_ports

    def __init__(self, decoder, log_softmax, **kwargs):
        TrainableNM.__init__(self, **kwargs)

        generator_params = {
            "max_sequence_length": self.local_parameters["max_seq_length"],
            "pad": self.local_parameters["pad_token"],
            "bos": self.local_parameters["bos_token"],
            "eos": self.local_parameters["eos_token"],
            "batch_size": self.local_parameters.get("batch_size", 1)
        }
        self.generator = GreedySequenceGenerator(
            decoder, log_softmax, **generator_params)

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
        max_sequence_length: maximum allowed length of generated sequences
        pad: index of padding token in the vocabulary
        bos: index of beginning of sequence token in the vocabulary
        eos: index of end of sequence token in the vocabulary
        device: torch.device to conduct generation on
        batch_size: size of the batch of generated sequences if no starting
            tokens are provided
        beam_size: size of the beam
        len_pen: parameter which penalizes shorter sequences
    """

    @staticmethod
    def create_ports():
        input_ports = {
            "hidden_states_src":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
            "input_mask_src":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            })
        }

        output_ports = {
            "output_ids":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            })
        }
        return input_ports, output_ports

    @property
    def num_weights(self):
        return 0

    def __init__(self, decoder, log_softmax, **kwargs):
        TrainableNM.__init__(self, **kwargs)

        params = self.local_parameters
        generator_params = {
            "max_sequence_length": params["max_seq_length"],
            "max_delta_length": params.get("max_delta_length", 50),
            "pad": params["pad_token"],
            "bos": params["bos_token"],
            "eos": params["eos_token"],
            "batch_size": params.get("batch_size", 1),
            "beam_size": params.get("beam_size", 4),
            "len_pen": params.get("length_penalty", 0)
        }
        self.generator = BeamSearchSequenceGenerator(
            decoder.embedding_layer, decoder.decoder, log_softmax,
            **generator_params)

    def forward(self, hidden_states_src, input_mask_src):
        output_ids = self.generator(
            encoder_hidden_states=hidden_states_src,
            encoder_input_mask=input_mask_src)
        return output_ids


class PaddedSmoothedCrossEntropyLossNM(LossNM):
    """
    Neural module which calculates CrossEntropyLoss and
    1) excludes padding tokens from loss calculation
    2) allows to use label smoothing regularization
    3) allows to calculate loss for the desired number of last tokens

    Args:
        label_smoothing: label smoothing regularization coefficient
        predict_last_k: how many last tokens to use for the loss calculation
    """

    @staticmethod
    def create_ports():
        input_ports = {
            "log_probs":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
            "target_ids":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
        }

        output_ports = {"loss": NeuralType(None)}
        return input_ports, output_ports

    def __init__(self, **kwargs):
        LossNM.__init__(self, **kwargs)

        loss_params = {
            "label_smoothing": self.local_parameters.get("label_smoothing", 0),
            "predict_last_k": self.local_parameters.get("predict_last_k", 0)
        }
        self._loss_fn = SmoothedCrossEntropyLoss(**loss_params)
        self._pad_id = self.local_parameters['pad_id']

    def _loss_function(self, log_probs, target_ids):
        target_mask = mask_padded_tokens(
            target_ids, self._pad_id).to(log_probs.dtype)
        loss = self._loss_fn(log_probs, target_ids, target_mask)
        return loss
