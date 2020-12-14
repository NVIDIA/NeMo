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
import numpy as np
import torch
import torch.nn as nn

from nemo.collections.tts.modules.fastspeech2_submodules import (
    FFTransformer,
    VariancePredictor,
    LengthRegulator,
    WaveformGenerator,
    WaveformDiscriminator,
)
from nemo.core.classes import NeuralModule, typecheck
from nemo.utils.decorators import experimental
from nemo.core.neural_types import *
from nemo.utils import logging


""" From the paper:
Our FastSpeech 2 consists of 4 feed-forward Transformer (FFT) blocks [20]
in the encoder and the mel-spectrogram decoder. In each FFT block, the dimension of phoneme
embeddings and the hidden size of the self-attention are set to 256. The number of attention heads
is set to 2 and the kernel sizes of the 1D-convolution in the 2-layer convolutional network after
the self-attention layer are set to 9 and 1, with input/output size of 256/1024 for the first layer and
1024/256 in the second layer. The output linear layer converts the 256-dimensional hidden states into
80-dimensional mel-spectrograms and optimized with mean absolute error (MAE). The size of the
phoneme vocabulary is 76, including punctuations. In the variance predictor, the kernel sizes of the
1D-convolution are set to 3, with input/output sizes of 256/256 for both layers and the dropout rate
is set to 0.5. Our waveform decoder consists of 1-layer transposed 1D-convolution with filter size
64 and 30 layers of dilated residual convolution blocks, whose skip channel size and kernel size of
1D-convolution are set to 64 and 3. The configurations of the discriminator in FastSpeech 2s are the
same as Parallel WaveGAN [27]. We list hyperparameters and configurations of all models used in
our experiments in Appendix A."""

# Very WIP.
# Hyperparams hard-coded for now to the best of my understanding


@experimental
class Encoder(NeuralModule):
    def __init__(self):
        """
        FastSpeech 2 encoder. Converts phoneme sequence to the phoneme hidden sequence.
        Consists of a phoneme embedding lookup, positional encoding, and four feed-forward
        Transformer blocks.

        Args:
        """
        # TODO: documentation of params
        super().__init__()

        self.encoder = FFTransformer(
            n_layer=4,
            n_head=2,
            d_model=256,
            d_head=256,
            d_inner=1024,
            kernel_size=(9, 1),
            dropout=0.2,
            dropatt=0.1,  # ??? Not sure if this is right, don't see it in paper
            embed_input=True,  # For the encoder, need to do embedding lookup
        )

    @property
    def input_types(self):  # phonemes
        return {"text": NeuralType(('B', 'T'), TokenIndex()), "text_lengths": NeuralType(('B'), LengthsType())}

    @property
    def output_types(self):
        return {
            "encoder_embedding": NeuralType(('B', 'T', 'D'), EmbeddedTextType()),
        }

    @typecheck()
    def forward(self, *, text, text_lengths):
        return self.encoder(text, seq_lens=text_lengths)


@experimental
class VarianceAdaptor(NeuralModule):
    def __init__(
        self,
        pitch=True,
        energy=True,
        max_duration=100,
        log_pitch=True,
        pitch_min=80.0,
        pitch_max=800.0,
        energy_min=0.0,
        energy_max=600.0,
    ):
        """
        FastSpeech 2 variance adaptor, which adds information like duration, pitch, etc. to the phoneme encoding.
        Sets of conv1D blocks with ReLU and dropout.

        Args:
        """
        # TODO: documentation of params
        super().__init__()
        # TODO: need to set all the other default min/max params - based on dataset?

        """In the variance predictor, the kernel sizes of the
        1D-convolution are set to 3, with input/output sizes of 256/256 for both layers and the dropout rate
        is set to 0.5."""
        # -- Duration Setup --
        # TODO: what should this max duration be? should this be set at all?
        self.max_duration = max_duration
        self.duration_predictor = VariancePredictor(d_model=256, d_inner=256, kernel_size=3, dropout=0.5)
        self.length_regulator = LengthRegulator()
        self.pitch = pitch
        self.energy = energy

        if self.pitch:
            if log_pitch:
                pitch_min = np.log(pitch_min)
                pitch_max = np.log(pitch_max)
            pitch_operator = torch.exp if log_pitch else lambda x: x
            pitch_bins = pitch_operator(torch.linspace(start=pitch_min, end=pitch_max, steps=255))
            # Prepend 0 for unvoiced frames
            pitch_bins = torch.cat((torch.tensor([0.0]), pitch_bins))

            # -- Pitch Setup --
            # NOTE: Pitch is clamped to 1e-5 which gets mapped to bin 1. But it is padded with 0s that get mapped to bin 0.
            self.register_buffer("pitch_bins", pitch_bins)
            self.pitch_predictor = VariancePredictor(
                d_model=256, d_inner=256, kernel_size=3, dropout=0.5  # va_hidden_size  # n_f0_bins
            )
            # Predictor outputs values directly rather than one-hot vectors, therefore Embedding
            self.pitch_lookup = nn.Embedding(256, 256)  # f0_bins, va_hidden_size

        if self.energy:
            # -- Energy Setup --
            self.register_buffer(  # Linear scale bins
                "energy_bins", torch.linspace(start=energy_min, end=energy_max, steps=255)  # n_energy_bins - 1
            )
            self.energy_predictor = VariancePredictor(
                d_model=256,
                d_inner=256,
                kernel_size=3,
                dropout=0.5,  # va_hidden_size, n_energy_bins, kernel size, dropout
            )
            self.energy_lookup = nn.Embedding(256, 256)  # n_energy_bins, va_hidden_size

    @property
    def input_types(self):
        return {
            "encoder_embedding": NeuralType(('B', 'T', 'D'), EmbeddedTextType()),
        }

    @property
    def output_types(self):
        return {
            # Might need a better name and type for this
            "variance_embedding": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }

    @typecheck()
    def forward(self, *, x, dur_target=None, pitch_target=None, energy_target=None):
        """
        Args:
            dur_target: Needs to be passed in during training. Duration targets for the duration predictor.
        """
        # TODO: no_grad instead of self.training?
        # TODO: or maybe condition on a new parameter like is_inference?

        # Duration predictions (or ground truth) fed into Length Regulator to
        # expand the hidden states of the encoder embedding
        dur_preds = self.duration_predictor(x.transpose(1, 2))
        # Output is Batch, Time
        if self.training:
            dur_out = self.length_regulator(x, dur_target)
        else:
            # dur_preds = torch.clamp(torch.round(torch.exp(dur_preds) - 1), min=0, max=self.max_duration)
            dur_preds = torch.clamp(torch.round(dur_preds), min=0).int()
            if not torch.sum(dur_preds, dim=1).bool().all():
                logging.error("Duration prediction failed on this batch. Settings to 1s")
                dur_preds += 1
            dur_out = self.length_regulator(x, dur_preds)
        out = dur_out

        pitch_preds = None
        if self.pitch:
            # Pitch
            # TODO: Add pitch spectrogram prediction & conversion back to pitch contour using iCWT
            #       (see Appendix C of the FastSpeech 2/2s paper).
            pitch_preds = self.pitch_predictor(dur_out.transpose(1, 2))
            if self.training:
                pitch_out = self.pitch_lookup(torch.bucketize(pitch_target, self.pitch_bins))
            else:
                pitch_out = self.pitch_lookup(torch.bucketize(pitch_preds, self.pitch_bins))
            out += pitch_out

        energy_preds = None
        if self.energy:
            # Energy
            energy_preds = self.energy_predictor(dur_out.transpose(1, 2))
            if self.training:
                energy_out = self.energy_lookup(torch.bucketize(energy_target, self.energy_bins))
            else:
                energy_out = self.energy_lookup(torch.bucketize(energy_preds, self.energy_bins))
            out += energy_out

        return out, dur_preds, pitch_preds, energy_preds


@experimental
class MelSpecDecoder(NeuralModule):
    def __init__(self):
        """
        FastSpeech 2 mel-spectrogram decoder. Converts adapted hidden sequence to a mel-spectrogram sequence.
        Consists of four feed-forward Transformer blocks.

        Args:
        """
        super().__init__()

        self.linear1 = nn.Linear(256, 384)  # Since input is 256
        self.decoder = FFTransformer(
            n_layer=4,
            n_head=2,
            d_model=384,  # Some paragraphs say 256, the table in the appendix says 384
            d_head=384,
            d_inner=1024,
            kernel_size=(9, 1),
            dropout=0.2,
            dropatt=0.1,  # ??? Not mentioned? Or am I just blind?
            embed_input=False,
        )
        self.linear2 = nn.Linear(384, 80)

    @property
    def input_types(self):
        # TODO
        pass

    @property
    def output_types(self):
        # TODO
        pass

    @typecheck()
    def forward(self, *, decoder_input, lengths):
        decoder_input = self.linear1(decoder_input)
        decoder_out, _ = self.decoder(decoder_input, lengths)
        mel_out = self.linear2(decoder_out)
        return mel_out


@experimental
class WaveformDecoder(NeuralModule):
    def __init__(
        self,
        in_channels=256,
        gen_out_channels=1,
        gen_trans_kernel_size=64,
        gen_hop_size=256,
        gen_n_layers=30,
        gen_dilation_cycle=3,
        gen_dilated_kernel_size=3,
        gen_residual_channels=64,
        gen_skip_channels=64,
        dis_out_channels=1,
        dis_n_layers=10,
        dis_kernel_size=3,
        dis_conv_channels=64,
        dis_conv_stride=1,
        dis_relu_alpha=0.2,
    ):
        """
        FastSpeech 2 waveform decoder. Converts adapted hidden sequence to a waveform sequence.
        Consists of one transposed conv1d layer, and 30 layers of dilated residual conv blocks.

        Args:
            in_channels (int): Number of input channels to the waveform decoder
            out_channels (int)
            gen_trans_kernel_size (int): Filter size of the upsampling transposed 1D convolution in the generator
            gen_n_layers (int): Number of layers of dilated residual convolutional blocks in the generator
            gen_dilation_cycle (int): The number of layers with a given dilation before moving up by a power of two.
                `n_layers` should be divisible by `dilation_cycle`.
            gen_dilated_kernel_size (int): Kernel size for the dilated conv1Ds in the generator
            gen_residual_channels (int)
            gen_skip_channels (int)
        """
        super().__init__()

        # WaveNet-based waveform generator
        self.generator = WaveformGenerator(
            in_channels=in_channels,
            out_channels=gen_out_channels,
            trans_kernel_size=gen_trans_kernel_size,
            hop_size=gen_hop_size,
            n_layers=gen_n_layers,
            dilation_cycle=gen_dilation_cycle,
            dilated_kernel_size=gen_dilated_kernel_size,
            residual_channels=gen_residual_channels,
            skip_channels=gen_skip_channels,
        )

        # Parallel WaveGAN-based discriminator
        self.discriminator = WaveformDiscriminator(
            in_channels=gen_out_channels,
            out_channels=dis_out_channels,
            n_layers=dis_n_layers,
            kernel_size=dis_kernel_size,
            conv_channels=dis_conv_channels,
            conv_stride=dis_conv_stride,
            relu_alpha=dis_relu_alpha,
        )

    @property
    def input_types(self):
        # TODO
        pass

    @property
    def output_types(self):
        # TODO
        pass

    @typecheck()
    def forward(self, *, decoder_input):
        generator_output = self.generator(decoder_input)
        pass
