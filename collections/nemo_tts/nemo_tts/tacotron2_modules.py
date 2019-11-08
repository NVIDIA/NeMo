# Copyright (c) 2019 NVIDIA Corporation
from functools import partial

import torch
from torch import nn
from torch.nn.functional import pad

from nemo.backends.pytorch.nm import TrainableNM, NonTrainableNM, LossNM
from nemo.core.neural_types import *
from .parts.tacotron2 import Encoder, Decoder, Postnet
from .parts.layers import get_mask_from_lengths


class TextEmbedding(TrainableNM):
    """ TODO: Docstring for Tacotron2Encdoer
    """
    @staticmethod
    def create_ports():
        input_ports = {
            "char_phone": NeuralType({0: AxisType(BatchTag),
                                      1: AxisType(TimeTag)})
        }

        output_ports = {
            "char_phone_embeddings": NeuralType({0: AxisType(BatchTag),
                                                 1: AxisType(EmbeddedTextTag),
                                                 2: AxisType(TimeTag)})
        }
        return input_ports, output_ports

    def __init__(self, n_symbols, symbols_embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(
            n_symbols, symbols_embedding_dim)
        self.to(self._device)

    def forward(self, char_phone):
        return self.embedding(char_phone).transpose(1, 2)


class Tacotron2Encoder(TrainableNM):
    """ TODO: Docstring for Tacotron2Encdoer
    """
    @staticmethod
    def create_ports():
        input_ports = {
            "char_phone_embeddings": NeuralType({0: AxisType(BatchTag),
                                                 1: AxisType(EmbeddedTextTag),
                                                 2: AxisType(TimeTag)}),
            "embedding_length": NeuralType({0: AxisType(BatchTag)})
        }

        output_ports = {
            "char_phone_encoded": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(EncodedRepresentationTag)})
        }
        return input_ports, output_ports

    def __init__(
            self,
            encoder_n_convolutions,
            encoder_embedding_dim,
            encoder_kernel_size,
            **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(encoder_n_convolutions=encoder_n_convolutions,
                               encoder_embedding_dim=encoder_embedding_dim,
                               encoder_kernel_size=encoder_kernel_size)
        self.to(self._device)

    def forward(self, char_phone_embeddings, embedding_length):
        char_phone_encoded = self.encoder(
            char_phone_embeddings, embedding_length)
        return char_phone_encoded


class Tacotron2Decoder(TrainableNM):
    """ TODO: Docstring for Tacotron2Decoder
    """
    @staticmethod
    def create_ports():
        input_ports = {
            "char_phone_encoded": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(EncodedRepresentationTag)}),
            "encoded_length": NeuralType({0: AxisType(BatchTag)}),
            "mel_target": NeuralType({0: AxisType(BatchTag),
                                      1: AxisType(MelSpectrogramSignalTag),
                                      2: AxisType(TimeTag)})
        }

        output_ports = {
            "mel_output": NeuralType({0: AxisType(BatchTag),
                                      1: AxisType(MelSpectrogramSignalTag),
                                      2: AxisType(TimeTag)}),
            "gate_output": NeuralType({0: AxisType(BatchTag),
                                       1: AxisType(TimeTag)}),
            "alignments": NeuralType({0: AxisType(BatchTag),
                                      1: AxisType(TimeTag),
                                      2: AxisType(TimeTag)})
        }
        return input_ports, output_ports

    def __init__(
            self,
            n_mel_channels,
            n_frames_per_step,
            encoder_embedding_dim,
            gate_threshold,
            prenet_dim,
            max_decoder_steps,
            decoder_rnn_dim,
            p_decoder_dropout,
            p_attention_dropout,
            attention_rnn_dim,
            attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size,
            **kwargs):
        super().__init__(**kwargs)
        self.collection = collection
        self.decoder = Decoder(
            n_mel_channels=n_mel_channels,
            n_frames_per_step=n_frames_per_step,
            encoder_embedding_dim=encoder_embedding_dim,
            gate_threshold=gate_threshold,
            prenet_dim=prenet_dim,
            max_decoder_steps=max_decoder_steps,
            decoder_rnn_dim=decoder_rnn_dim,
            p_decoder_dropout=p_decoder_dropout,
            p_attention_dropout=p_attention_dropout,
            attention_rnn_dim=attention_rnn_dim,
            attention_dim=attention_dim,
            attention_location_n_filters=attention_location_n_filters,
            attention_location_kernel_size=attention_location_kernel_size)
        self.to(self._device)

    def forward(self, char_phone_encoded, encoded_length, mel_target):
        if self.training:
            mel_output, gate_output, alignments = self.decoder(
                char_phone_encoded, mel_target, memory_lengths=encoded_length)
        else:
            mel_output, gate_output, alignments, _ = self.decoder.infer(
                char_phone_encoded, memory_lengths=encoded_length)
        return mel_output, gate_output, alignments


class Tacotron2DecoderInfer(Tacotron2Decoder):
    """ TODO: Docstring for Tacotron2Decoder
    """
    @staticmethod
    def create_ports():
        input_ports = {
            "char_phone_encoded": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(EncodedRepresentationTag)}),
            "encoded_length": NeuralType({0: AxisType(BatchTag)}),
            "mel_target": NeuralType({0: AxisType(BatchTag),
                                      1: AxisType(MelSpectrogramSignalTag),
                                      2: AxisType(TimeTag)})
        }

        output_ports = {
            "mel_output": NeuralType({0: AxisType(BatchTag),
                                      1: AxisType(MelSpectrogramSignalTag),
                                      2: AxisType(TimeTag)}),
            "gate_output": NeuralType({0: AxisType(BatchTag),
                                       1: AxisType(TimeTag)}),
            "alignments": NeuralType({0: AxisType(BatchTag),
                                      1: AxisType(TimeTag),
                                      2: AxisType(TimeTag)}),
            "mel_len": NeuralType({0: AxisType(BatchTag)})
        }
        return input_ports, output_ports

    def __str__(self):
        return "Tacotron2Decoder"

    def __init__(
            self,
            n_mel_channels,
            n_frames_per_step,
            encoder_embedding_dim,
            gate_threshold,
            prenet_dim,
            max_decoder_steps,
            decoder_rnn_dim,
            p_decoder_dropout,
            p_attention_dropout,
            attention_rnn_dim,
            attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size,
            **kwargs):
        super().__init__(
            n_mel_channels,
            n_frames_per_step,
            encoder_embedding_dim,
            gate_threshold,
            prenet_dim,
            max_decoder_steps,
            decoder_rnn_dim,
            p_decoder_dropout,
            p_attention_dropout,
            attention_rnn_dim,
            attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size,
            **kwargs)

    def forward(self, char_phone_encoded, encoded_length, mel_target):
        if self.training:
            raise ValueError("You are using the Tacotron 2 Infer Neural Module"
                             " in training mode.")
        with torch.no_grad():
            mel_output, gate_output, alignments, mel_len = self.decoder.infer(
                char_phone_encoded, memory_lengths=encoded_length)
        return mel_output, gate_output, alignments, mel_len


class Tacotron2Postnet(TrainableNM):
    """ TODO: Docstring for Tacotron2Postnet
    """
    @staticmethod
    def create_ports():
        input_ports = {
            "mel_input": NeuralType({0: AxisType(BatchTag),
                                     1: AxisType(MelSpectrogramSignalTag),
                                     2: AxisType(TimeTag)}),
        }

        output_ports = {
            "mel_output": NeuralType({0: AxisType(BatchTag),
                                      1: AxisType(MelSpectrogramSignalTag),
                                      2: AxisType(TimeTag)}),
        }
        return input_ports, output_ports

    def __init__(
            self,
            n_mel_channels,
            postnet_embedding_dim,
            postnet_kernel_size,
            postnet_n_convolutions,
            **kwargs):
        super().__init__(**kwargs)
        self.postnet = Postnet(n_mel_channels=n_mel_channels,
                               postnet_embedding_dim=postnet_embedding_dim,
                               postnet_kernel_size=postnet_kernel_size,
                               postnet_n_convolutions=postnet_n_convolutions)
        self.to(self._device)

    def forward(self, mel_input):
        return self.postnet(mel_input) + mel_input


class Tacotron2Loss(LossNM):
    """TODO
    """
    @staticmethod
    def create_ports():
        input_ports = {
            "mel_out": NeuralType({0: AxisType(BatchTag),
                                   1: AxisType(MelSpectrogramSignalTag),
                                   2: AxisType(TimeTag)}),
            "mel_out_postnet": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(MelSpectrogramSignalTag),
                2: AxisType(TimeTag)}),
            "gate_out": NeuralType({0: AxisType(BatchTag),
                                    1: AxisType(TimeTag)}),
            "mel_target": NeuralType({0: AxisType(BatchTag),
                                      1: AxisType(MelSpectrogramSignalTag),
                                      2: AxisType(TimeTag)}),
            "gate_target": NeuralType({0: AxisType(BatchTag),
                                      1: AxisType(TimeTag)}),
            "target_len": NeuralType({0: AxisType(BatchTag)}),
            "seq_len": NeuralType({0: AxisType(BatchTag)}),
        }

        output_ports = {"loss": NeuralType(None)}
        return input_ports, output_ports

    def __init__(self, pad_value=-11.52, **kwargs):
        super().__init__(**kwargs)
        self.pad_value = pad_value

    def _loss_function(self, **kwargs):
        return self._loss(*(kwargs.values()))

    def _loss(self, mel_out, mel_out_postnet, gate_out,
              mel_target, gate_target, target_len, seq_len):
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        max_len = torch.max(target_len).item()
        max_pad = (16 - (max_len % 16)) % 16
        max_len += max_pad

        if max_len < mel_out.shape[2]:
            # Predicted len is larger than reference
            # Need to slice
            mel_out = mel_out.narrow(2, 0, max_len)
            mel_out_postnet = mel_out_postnet.narrow(2, 0, max_len)
            gate_out = gate_out.narrow(1, 0, max_len).contiguous()
        elif max_len > mel_out.shape[2]:
            # Need to do padding
            pad_amount = max_len - mel_out.shape[2]
            mel_out = pad(mel_out, (0, pad_amount), value=self.pad_value)
            mel_out_postnet = pad(
                mel_out_postnet, (0, pad_amount), value=self.pad_value)
            gate_out = pad(gate_out, (0, pad_amount), value=1e3)
            max_len = mel_out.shape[2]

        mask = ~get_mask_from_lengths(target_len, max_len=max_len)
        mask = mask.expand(mel_target.shape[1], mask.size(0), mask.size(1))
        mask = mask.permute(1, 0, 2)
        mel_out.data.masked_fill_(mask, self.pad_value)
        mel_out_postnet.data.masked_fill_(mask, self.pad_value)
        gate_out.data.masked_fill_(mask[:, 0, :], 1e3)

        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss


class MakeGate(NonTrainableNM):
    """TODO
    """
    @staticmethod
    def create_ports():
        input_ports = {
            "target_len": NeuralType({0: AxisType(BatchTag)}),
        }

        output_ports = {
            "gate_target": NeuralType({0: AxisType(BatchTag),
                                       1: AxisType(TimeTag)})
        }
        return input_ports, output_ports

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, target_len):
        max_len = torch.max(target_len)
        max_pad = (16 - (max_len % 16)) % 16
        max_len += max_pad
        gate_padded = torch.FloatTensor(target_len.shape[0], max_len)
        gate_padded.zero_()
        for i, length in enumerate(target_len):
            gate_padded[i, length.data-1:] = 1
        return gate_padded.to(device=self._device)
