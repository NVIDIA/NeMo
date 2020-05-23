# Copyright (c) 2019 NVIDIA Corporation
from functools import partial

import torch
from torch import nn
from torch.nn.functional import pad

from nemo import logging
from nemo.backends.pytorch.nm import LossNM, NonTrainableNM, TrainableNM
from nemo.collections.tts.parts.layers import get_mask_from_lengths
from nemo.collections.tts.parts.tacotron2 import Decoder, Encoder, Postnet
from nemo.core.neural_types import *
from nemo.utils.decorators import add_port_docs

__all__ = [
    "MakeGate",
    "Tacotron2Loss",
    "Tacotron2Postnet",
    "Tacotron2Decoder",
    "Tacotron2DecoderInfer",
    "Tacotron2Encoder",
    "TextEmbedding",
]


class TextEmbedding(TrainableNM):
    """
    TextEmbedding embeds the encoded character labels to an embedding space

    Args:
        n_symbols (int): The number of character labels. The input char_phone's
            second axis dim size should be n_symbols.
        symbols_embedding_dim (int): The size of the embedding dimension.
            Defaults to 512.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        # return {"char_phone": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)})}
        return {"char_phone": NeuralType(('B', 'T'), LabelsType())}

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # "char_phone_embeddings": NeuralType(
            #     {0: AxisType(BatchTag), 1: AxisType(EmbeddedTextTag), 2: AxisType(TimeTag),}
            # )
            "char_phone_embeddings": NeuralType(('B', 'D', 'T'), EmbeddedTextType())
        }

    def __init__(self, n_symbols, symbols_embedding_dim: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, symbols_embedding_dim)
        self.to(self._device)

    def forward(self, char_phone):
        return self.embedding(char_phone).transpose(1, 2)


class Tacotron2Encoder(TrainableNM):
    """
    Tacotron2Encoder is the encoder part of Tacotron 2. It takes embedded text
    as input and creates an encoded representation of the text that can be used
    with downstream attention and decoders.

    Args:
        encoder_n_convolutions (int): The number of convolution layers inside
            the encoder. Defaults to 5.
        encoder_embedding_dim (int): The size of the embedded text. It will
            also be the output size of the encoded text. Defaults to 512.
        encoder_kernel_size (int): The kernel size of the convolution layers.
            Defaults to 3
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "char_phone_embeddings": NeuralType(
            #     {0: AxisType(BatchTag), 1: AxisType(EmbeddedTextTag), 2: AxisType(TimeTag),}
            # ),
            # "embedding_length": NeuralType({0: AxisType(BatchTag)}),
            "char_phone_embeddings": NeuralType(('B', 'D', 'T'), EmbeddedTextType()),
            "embedding_length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # "char_phone_encoded": NeuralType(
            #     {0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(EncodedRepresentationTag),}
            # )
            "char_phone_encoded": NeuralType(('B', 'T', 'D'), EncodedRepresentation())
        }

    def __init__(
        self, encoder_n_convolutions: int = 5, encoder_embedding_dim: int = 512, encoder_kernel_size: int = 3
    ):
        super().__init__()
        self.encoder = Encoder(
            encoder_n_convolutions=encoder_n_convolutions,
            encoder_embedding_dim=encoder_embedding_dim,
            encoder_kernel_size=encoder_kernel_size,
        )
        self.to(self._device)

    def forward(self, char_phone_embeddings, embedding_length):
        char_phone_encoded = self.encoder(char_phone_embeddings, embedding_length)
        return char_phone_encoded


class Tacotron2Decoder(TrainableNM):
    """
    Tacotron2Decoder implements the attention, decoder, and prenet parts of
    Tacotron 2. It takes the encoded text and produces mel spectrograms. The
    decoder contains two rnns, one is called the decoder rnn and the other is
    called the attention rnn.

    Args:
        n_mel_channels (int): The size or dimensionality of the mel spectrogram
        n_frames_per_step (int): The number of frames we predict at each
            decoder time step. Defaults to 1
        encoder_embedding_dim (int): The size of the encoded text.
            Defaults to 512.
        gate_threshold (float): A number in [0, 1). When teacher forcing is
            not used, the model predict a stopping value at each model time
            step. The model will stop if the value is greater than
            gate_threshold. Defaults to 0.5.
        prenet_dim (int): The hidden dimension of the prenet. Defaults to 256.
        max_decoder_steps (int): When not teacher forcing, the maximum number
            of frames to predict. Defaults to 1000.
        decoder_rnn_dim (int): The hidden dimension of the decoder rnn.
            Defaults to 1024.
        p_decoder_dropout (float): Dropout probability for the decoder rnn.
            Defaults to 0.1.
        p_attention_dropout (float): Dropout probability for the attention rnn.
            Defaults to 0.1.
        attention_rnn_dim (int): The hidden dimension of the attention rnn.
            Defaults to 1024.
        attention_dim (int): The hidden dimension of the attention mechanism.
            Defaults to 128.
        attention_location_n_filters (int): The number of convolution filters
            for the location part of the attention mechanism.
            Defaults to 32.
        attention_location_kernel_size (int): The kernel size of the
            convolution for the location part of the attention mechanism.
            Defaults to 31.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "char_phone_encoded": NeuralType(
            #     {0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(EncodedRepresentationTag),}
            # ),
            # "encoded_length": NeuralType({0: AxisType(BatchTag)}),
            # "mel_target": NeuralType(
            #     {0: AxisType(BatchTag), 1: AxisType(MelSpectrogramSignalTag), 2: AxisType(TimeTag),}
            # ),
            "char_phone_encoded": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "encoded_length": NeuralType(tuple('B'), LengthsType()),
            "mel_target": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # "mel_output": NeuralType(
            #     {0: AxisType(BatchTag), 1: AxisType(MelSpectrogramSignalTag), 2: AxisType(TimeTag),}
            # ),
            # "gate_output": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "alignments": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(TimeTag),}),
            "mel_output": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "gate_output": NeuralType(('B', 'T'), ChannelType()),
            "alignments": NeuralType(('B', 'T', 'T'), ChannelType()),
        }

    def __init__(
        self,
        n_mel_channels: int,
        n_frames_per_step: int = 1,
        encoder_embedding_dim: int = 512,
        gate_threshold: float = 0.5,
        prenet_dim: int = 256,
        max_decoder_steps: int = 1000,
        decoder_rnn_dim: int = 1024,
        p_decoder_dropout: float = 0.1,
        p_attention_dropout: float = 0.1,
        attention_rnn_dim: int = 1024,
        attention_dim: int = 128,
        attention_location_n_filters: int = 32,
        attention_location_kernel_size: int = 31,
        prenet_p_dropout: float = 0.5,
        force: bool = False,
    ):
        super().__init__()
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
            attention_location_kernel_size=attention_location_kernel_size,
            prenet_p_dropout=prenet_p_dropout,
            early_stopping=True,
        )
        self.force = force
        self.to(self._device)

    def forward(self, char_phone_encoded, encoded_length, mel_target):
        if self.training or self.force:
            mel_output, gate_output, alignments = self.decoder(
                char_phone_encoded, mel_target, memory_lengths=encoded_length
            )
        else:
            mel_output, gate_output, alignments, _ = self.decoder.infer(
                char_phone_encoded, memory_lengths=encoded_length
            )
        return mel_output, gate_output, alignments


class Tacotron2DecoderInfer(Tacotron2Decoder):
    """
    Tacotron2DecoderInfer is an inference Neural Module used in place
    of the Tacotron2Decoder NM.

    Args:
        n_mel_channels (int): The size or dimensionality of the mel spectrogram
        n_frames_per_step (int): The number of frames we predict at each
            decoder time step. Defaults to 1
        encoder_embedding_dim (int): The size of the encoded text.
            Defaults to 512.
        gate_threshold (float): A number in [0, 1). When teacher forcing is
            not used, the model predict a stopping value at each model time
            step. The model will stop if the value is greater than
            gate_threshold. Defaults to 0.5.
        prenet_dim (int): The hidden dimension of the prenet. Defaults to 256.
        max_decoder_steps (int): When not teacher forcing, the maximum number
            of frames to predict. Defaults to 1000.
        decoder_rnn_dim (int): The hidden dimension of the decoder rnn.
            Defaults to 1024.
        p_decoder_dropout (float): Dropout probability for the decoder rnn.
            Defaults to 0.1.
        p_attention_dropout (float): Dropout probability for the attention rnn.
            Defaults to 0.1.
        attention_rnn_dim (int): The hidden dimension of the attention rnn.
            Defaults to 1024.
        attention_dim (int): The hidden dimension of the attention mechanism.
            Defaults to 128.
        attention_location_n_filters (int): The number of convolution filters
            for the location part of the attention mechanism.
            Defaults to 32.
        attention_location_kernel_size (int): The kernel size of the
            convolution for the location part of the attention mechanism.
            Defaults to 31.
    """

    def __init__(
        self,
        n_mel_channels: int,
        n_frames_per_step: int = 1,
        encoder_embedding_dim: int = 512,
        gate_threshold: float = 0.5,
        prenet_dim: int = 256,
        max_decoder_steps: int = 1000,
        decoder_rnn_dim: int = 1024,
        p_decoder_dropout: float = 0.1,
        p_attention_dropout: float = 0.1,
        attention_rnn_dim: int = 1024,
        attention_dim: int = 128,
        attention_location_n_filters: int = 32,
        attention_location_kernel_size: int = 31,
        prenet_p_dropout: float = 0.5,
        force: bool = False,
    ):
        super().__init__(
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
            attention_location_kernel_size=attention_location_kernel_size,
            prenet_p_dropout=prenet_p_dropout,
            force=force,
        )

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "char_phone_encoded": NeuralType(
            #     {0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(EncodedRepresentationTag),}
            # ),
            # "encoded_length": NeuralType({0: AxisType(BatchTag)}),
            "char_phone_encoded": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "encoded_length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # "mel_output": NeuralType(
            #     {0: AxisType(BatchTag), 1: AxisType(MelSpectrogramSignalTag), 2: AxisType(TimeTag),}
            # ),
            # "gate_output": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "alignments": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(TimeTag),}),
            # "mel_len": NeuralType({0: AxisType(BatchTag)}),
            "mel_output": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "gate_output": NeuralType(('B', 'T'), ChannelType()),
            "alignments": NeuralType(('B', 'T', 'T'), ChannelType()),
            "mel_len": NeuralType(tuple('B'), LengthsType()),
        }

    def __str__(self):
        return "Tacotron2Decoder"

    def forward(self, char_phone_encoded, encoded_length):
        if self.training:
            raise ValueError("You are using the Tacotron 2 Infer Neural Module in training mode.")
        with torch.no_grad():
            mel_output, gate_output, alignments, mel_len = self.decoder.infer(
                char_phone_encoded, memory_lengths=encoded_length
            )
        return mel_output, gate_output, alignments, mel_len


class Tacotron2Postnet(TrainableNM):
    """
    Tacotron2Postnet implements the postnet part of Tacotron 2. It takes a mel
    spectrogram as generated by the decoder and corrects errors within the
    generated mel spectrogram.

    Args:
        n_mel_channels (int): The size or dimensionality of the mel spectrogram
        postnet_embedding_dim (int): Hidden size of convolutions.
            Defaults to 512.
        postnet_kernel_size (int): Kernel size of convolutions.
            Defaults to 5.
        postnet_n_convolutions (int): Number of convolution layers.
            Defaults to 5.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "mel_input": NeuralType(
            #     {0: AxisType(BatchTag), 1: AxisType(MelSpectrogramSignalTag), 2: AxisType(TimeTag),}
            # )
            "mel_input": NeuralType(('B', 'D', 'T'), MelSpectrogramType())
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # "mel_output": NeuralType(
            #     {0: AxisType(BatchTag), 1: AxisType(MelSpectrogramSignalTag), 2: AxisType(TimeTag),}
            # ),
            "mel_output": NeuralType(('B', 'D', 'T'), MelSpectrogramType())
        }

    def __init__(
        self,
        n_mel_channels: int,
        postnet_embedding_dim: int = 512,
        postnet_kernel_size: int = 5,
        postnet_n_convolutions: int = 5,
        p_dropout: float = 0.5,
    ):
        super().__init__()
        self.postnet = Postnet(
            n_mel_channels=n_mel_channels,
            postnet_embedding_dim=postnet_embedding_dim,
            postnet_kernel_size=postnet_kernel_size,
            postnet_n_convolutions=postnet_n_convolutions,
            p_dropout=p_dropout,
        )
        self.to(self._device)

    def forward(self, mel_input):
        return self.postnet(mel_input) + mel_input


class Tacotron2Loss(LossNM):
    """
    Tacoton2Loss implements the loss function of Tacotron 2. The loss function
    is the mean squared error between the reference mel spectrogram and the
    mel spectrogram predicted by the decoder + the mean squared error between
    the reference mel spectrogram and the mel spectrogram predicted by the
    post net + the cross entropy error between the stop values and the
    reference mel length.

    Args:
        pad_value (float): In the evaluation case, when we don't use teacher
            forcing, if the generated mel is shorter than the reference mel,
            we pad the generated mel with this value. Default is ~log(1e-5).
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "mel_out": NeuralType(
            #     {0: AxisType(BatchTag), 1: AxisType(MelSpectrogramSignalTag), 2: AxisType(TimeTag),}
            # ),
            # "mel_out_postnet": NeuralType(
            #     {0: AxisType(BatchTag), 1: AxisType(MelSpectrogramSignalTag), 2: AxisType(TimeTag),}
            # ),
            # "gate_out": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "mel_target": NeuralType(
            #     {0: AxisType(BatchTag), 1: AxisType(MelSpectrogramSignalTag), 2: AxisType(TimeTag),}
            # ),
            # "gate_target": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "target_len": NeuralType({0: AxisType(BatchTag)}),
            # "seq_len": NeuralType({0: AxisType(BatchTag)}),
            "mel_out": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "mel_out_postnet": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "gate_out": NeuralType(('B', 'T'), ChannelType()),
            "mel_target": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "gate_target": NeuralType(('B', 'T'), ChannelType()),
            "target_len": NeuralType(tuple('B'), LengthsType()),
            "seq_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, pad_value: float = -11.52):
        super().__init__()
        self.pad_value = pad_value

    def _loss_function(self, **kwargs):
        return self._loss(*(kwargs.values()))

    def _loss(
        self, mel_out, mel_out_postnet, gate_out, mel_target, gate_target, target_len, seq_len,
    ):
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        max_len = mel_target.shape[2]

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
            mel_out_postnet = pad(mel_out_postnet, (0, pad_amount), value=self.pad_value)
            gate_out = pad(gate_out, (0, pad_amount), value=1e3)
            max_len = mel_out.shape[2]

        mask = ~get_mask_from_lengths(target_len, max_len=max_len)
        mask = mask.expand(mel_target.shape[1], mask.size(0), mask.size(1))
        mask = mask.permute(1, 0, 2)
        mel_out.data.masked_fill_(mask, self.pad_value)
        mel_out_postnet.data.masked_fill_(mask, self.pad_value)
        gate_out.data.masked_fill_(mask[:, 0, :], 1e3)

        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss


class MakeGate(NonTrainableNM):
    """MakeGate is a helper Neural Module that makes the target stop value.
    """

    def __init__(self):
        super().__init__()

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "target_len": NeuralType({0: AxisType(BatchTag)}),
            # "mel_target": NeuralType(
            #     {0: AxisType(BatchTag), 1: AxisType(MelSpectrogramSignalTag), 2: AxisType(TimeTag),}
            # ),
            "target_len": NeuralType(tuple('B'), LengthsType()),
            "mel_target": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        # return {"gate_target": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)})}
        return {"gate_target": NeuralType(('B', 'T'), ChannelType())}

    def forward(self, target_len, mel_target):
        max_len = mel_target.shape[2]
        gate_padded = torch.FloatTensor(target_len.shape[0], max_len)
        gate_padded.zero_()
        for i, length in enumerate(target_len):
            gate_padded[i, length.data - 1 :] = 1
        return gate_padded.to(device=self._device)
