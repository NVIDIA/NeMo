# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
    LengthRegulator,
    LengthRegulator2,
    VariancePredictor,
    WaveformDiscriminator,
    WaveformGenerator,
)
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import *
from nemo.utils import logging
from nemo.utils.decorators import experimental


@experimental
class FastSpeech2Encoder(NeuralModule):
    def __init__(
        self,
        d_model=256,
        n_layers=4,
        n_attn_heads=2,
        d_attn_head=256,
        d_inner=1024,
        kernel_size=9,
        dropout=0.1,
        attn_dropout=0.1,
        n_embed=84,
        padding_idx=83,
    ):
        """
        FastSpeech 2 encoder. Converts phoneme sequence to the phoneme hidden sequence.
        Consists of a phoneme embedding lookup, positional encoding, and feed-forward
        Transformer blocks (4 by default).

        Args:
            d_model: Model input (embedding) dimension. Defaults to 256.
            n_layers: Number of feed-forward Transformer layers in the encoder. Defaults to 4.
            n_attn_heads: Number of attention heads for the feed-forward Transformer in the encoder.
                Defaults to 2.
            d_attn_head: Dimensionality of the attention heads. Defaults to 256.
            d_inner: Encoder hidden dimension. Defaults to 1024.
            kernel_size: Encoder Conv1d kernel size (kernel_size, 1). Defaults to 9.
            dropout: Encoder feed-forward Transformer dropout. Defaults to 0.1.
            attn_dropout: Encoder attention dropout. Defaults to 0.1.
            n_embed: Embedding input dim, should match number of tokens. Defaults to 84.
            padding_idx: Padding token index. Deafaults to 83.
        """
        super().__init__()

        self.encoder = FFTransformer(
            n_layer=n_layers,
            n_head=n_attn_heads,
            d_model=d_model,
            d_head=d_attn_head,
            d_inner=d_inner,
            kernel_size=(kernel_size, 1),
            dropout=dropout,
            dropatt=attn_dropout,  # Not in the FS2 paper
            embed_input=True,  # For the encoder, need to do embedding lookup
            n_embed=n_embed,
            padding_idx=padding_idx
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
        d_model=256,
        dropout=0.2,
        dur_d_hidden=256,
        dur_kernel_size=3,
        max_duration=100,
        va_hidden=256,
        pitch=True,
        log_pitch=True,
        n_f0_bins=256,
        pitch_kernel_size=3,
        pitch_min=80.0,
        pitch_max=800.0,
        energy=True,
        n_energy_bins=256,
        energy_kernel_size=3,
        energy_min=0.0,
        energy_max=600.0,
        vocab=None,
        # use_guassian_embed=False,
    ):
        """
        FastSpeech 2 variance adaptor, which adds information like duration, pitch, etc. to the phoneme encoding.
        Sets of conv1D blocks with ReLU and dropout.

        Args:
            d_model: Input and hidden dimension. Defaults to 256 (default encoder output dim).
            dropout: Variance adaptor dropout. Defaults to 0.2.
            dur_d_hidden: Hidden dim of the duration predictor. Defaults to 256.
            dur_kernel_size: Kernel size for the duration predictor. Defaults to 3.
            max_duration: ### Currently unused ###
            va_hidden: Hidden dimension of the variance adaptor, i.e. the output of the pitch predictor and
                energy predictor. Defaults to 256.
            pitch (bool): Whether or not to use the pitch predictor.
            log_pitch (bool): If True, uses log pitch. Defaults to True.
            n_f0_bins: Number of F0 bins for the pitch predictor. Defaults to 256.
            pitch_kernel_size: Kernel size for the pitch predictor. Defaults to 3.
            pitch_min: Defaults to 80.0.
            pitch_max: Defaults to 800.0.
            pitch_d_hidden: Hidden dim of the pitch predictor.
            energy (bool): Whether or not to use the energy predictor.
            n_energy_bins: Number of energy bins. Defaults to 256.
            energy_kernel_size: Kernel size for the energy predictor. Defaults to 3.
            energy_min: Defaults to 0.0.
            energy_max: Defaults to 600.0.
        """
        super().__init__()

        # -- Duration Setup --
        # TODO: what should this max duration be? should this be set at all? (Not currently used.)
        self.max_duration = max_duration
        self.duration_predictor = VariancePredictor(
            d_model=d_model, d_inner=dur_d_hidden, kernel_size=dur_kernel_size, dropout=dropout
        )
        if vocab is not None:
            self.length_regulator = GaussianEmbedding(vocab)
        else:
            self.length_regulator = LengthRegulator2()

        self.pitch = pitch
        self.energy = energy

        # -- Pitch Setup --
        # NOTE: Pitch is clamped to 1e-5 which gets mapped to bin 1. But it is padded with 0s that get mapped to bin 0.
        if self.pitch:
            if log_pitch:
                pitch_min = np.log(pitch_min)
                pitch_max = np.log(pitch_max)
            pitch_operator = torch.exp if log_pitch else lambda x: x
            pitch_bins = pitch_operator(torch.linspace(start=pitch_min, end=pitch_max, steps=n_f0_bins - 1))
            # Prepend 0 for unvoiced frames
            pitch_bins = torch.cat((torch.tensor([0.0]), pitch_bins))

            self.register_buffer("pitch_bins", pitch_bins)
            self.pitch_predictor = VariancePredictor(
                d_model=d_model, d_inner=n_f0_bins, kernel_size=pitch_kernel_size, dropout=dropout
            )
            # Predictor outputs values directly rather than one-hot vectors, therefore Embedding
            self.pitch_lookup = nn.Embedding(n_f0_bins, d_model)

        # -- Energy Setup --
        if self.energy:
            self.register_buffer(  # Linear scale bins
                "energy_bins", torch.linspace(start=energy_min, end=energy_max, steps=n_energy_bins - 1)
            )
            self.energy_predictor = VariancePredictor(
                d_model=d_model, d_inner=n_energy_bins, kernel_size=energy_kernel_size, dropout=dropout,
            )
            self.energy_lookup = nn.Embedding(n_energy_bins, d_model)

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
    def forward(self, *, x, x_len, dur_target=None, pitch_target=None, energy_target=None, spec_len=None):
        """
        Args:
            dur_target: Needs to be passed in during training. Duration targets for the duration predictor.
        """
        # TODO: no_grad instead of self.training?
        # TODO: or maybe condition on a new parameter like is_inference?

        # Duration predictions (or ground truth) fed into Length Regulator to
        # expand the hidden states of the encoder embedding
        log_dur_preds = self.duration_predictor(x)
        log_dur_preds.masked_fill_(~get_mask_from_lengths(x_len), 0)
        # Output is Batch, Time
        if dur_target is not None:
            dur_out = self.length_regulator(x, dur_target)
        else:
            dur_preds = torch.clamp_min(torch.round(torch.exp(log_dur_preds)) - 1, 0).long()
            if not torch.sum(dur_preds, dim=1).bool().all():
                logging.error("Duration prediction failed on this batch. Settings to 1s")
                dur_preds += 1
            dur_out = self.length_regulator(x, dur_preds)
            spec_len = torch.sum(dur_preds, dim=1)
        out = dur_out
        out *= get_mask_from_lengths(spec_len).unsqueeze(-1)

        # Pitch
        # TODO: Add pitch spectrogram prediction & conversion back to pitch contour using iCWT
        #       (see Appendix C of the FastSpeech 2/2s paper).
        pitch_preds = None
        if self.pitch:
            pitch_preds = self.pitch_predictor(dur_out)
            pitch_preds.masked_fill_(~get_mask_from_lengths(spec_len), 0)
            if pitch_target is not None:
                pitch_out = self.pitch_lookup(torch.bucketize(pitch_target, self.pitch_bins))
            else:
                # pitch_preds = torch.clamp_min(torch.exp(lg_pitch_preds) - 1, 0)
                pitch_out = self.pitch_lookup(torch.bucketize(pitch_preds, self.pitch_bins))
            out += pitch_out
        out *= get_mask_from_lengths(spec_len).unsqueeze(-1)

        # Energy
        energy_preds = None
        if self.energy:
            energy_preds = self.energy_predictor(dur_out)
            if energy_target is not None:
                energy_out = self.energy_lookup(torch.bucketize(energy_target, self.energy_bins))
            else:
                energy_out = self.energy_lookup(torch.bucketize(energy_preds, self.energy_bins))
            out += energy_out
        out *= get_mask_from_lengths(spec_len).unsqueeze(-1)

        return out, log_dur_preds, pitch_preds, energy_preds, spec_len


class GaussianEmbedding(nn.Module):
    """Gaussian embedding layer."""

    EPS = 1e-6

    def __init__(self, vocab, sigma_c=2.0):
        super().__init__()

        self.pad = vocab.pad
        self.sigma_c = sigma_c

    def forward(self, text, durs):
        """
        durs : Shape: Batch x Text Length
        """
        total_time = torch.max(durs.sum(1))

        # Calculate mean from durations
        mean = (durs / 2.0) + torch.cumsum(durs, dim=-1)  # B x SL
        mean = mean.unsqueeze(1).repeat(1, total_time, 1)  # B X TL X SL

        # Calculate sigma from durations
        sigmas = durs
        sigmas = sigmas.float() / self.sigma_c
        sigmas = sigmas.unsqueeze(1).repeat(1, total_time, 1) + self.EPS
        assert mean.shape == sigmas.shape

        # Times indexes
        t_ind = torch.arange(total_time, device=mean.device).view(1, -1, 1).repeat(durs.shape[0], 1, durs.shape[-1])
        t_ind = t_ind.float() + 0.5

        # Weights: [B,T,N]
        distribution = torch.distributions.normal.Normal(mean, sigmas)
        probs = distribution.log_prob(t_ind).exp()

        # Normalize distribution per spectorgram frame
        probs = probs / (probs.sum(-1, keepdim=True) + self.EPS)
        return torch.matmul(probs, text)


@experimental
class MelSpecDecoder(NeuralModule):
    def __init__(
        self,
        d_model=256,
        d_out=80,
        n_layers=4,
        n_attn_heads=2,
        d_attn_head=256,
        d_inner=1024,
        kernel_size=9,
        dropout=0.1,
        attn_dropout=0.1,
    ):
        """
        FastSpeech 2 mel-spectrogram decoder. Converts adapted hidden sequence to a mel-spectrogram sequence.
        Consists of four feed-forward Transformer blocks.

        Args:
            d_model: Input dimension. Defaults to 256.
            d_out: Dimensionality of output. Defaults to 80.
            n_layers: Number of feed-forward Transformer layers in the mel-spec decoder. Defaults to 4.
            n_attn_heads: Number of attention heads for the feed-forward Transformer. Defaults to 2.
            d_attn_head: Dimensionality of the attention heads. Defaults to 256.
            d_inner: Mel-spec decoder hidden dimension. Defaults to 1024.
            kernel_size: Mel-spec decoder Conv1d kernel size (kernel_size, 1). Defaults to 9.
            dropout: Mel-spec decoder feed-forward Transformer dropout. Defaults to 0.1.
            attn_dropout: Mel-spec decoder attention dropout. Defaults to 0.1.
        """
        super().__init__()

        self.decoder = FFTransformer(
            n_layer=n_layers,
            n_head=n_attn_heads,
            d_model=d_model,
            d_head=d_attn_head,
            d_inner=d_inner,
            kernel_size=(kernel_size, 1),
            dropout=dropout,
            dropatt=attn_dropout,
            embed_input=False,
        )
        self.linear = nn.Linear(d_model, d_out)

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
        decoder_out, _ = self.decoder(decoder_input, lengths)
        mel_out = self.linear(decoder_out)
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
            in_channels (int): Number of input channels to the waveform decoder. Defaults to 256.
            gen_out_channels (int): Number of output channels of the waveform decoder generator. Defaults to 1.
            gen_trans_kernel_size (int): Filter size of the upsampling transposed 1D convolution in the generator.
                Defaults to 64.
            gen_hop_size (int): Hop size for input upsampling in the waveform generator. Defaults to 256.
            gen_n_layers (int): Number of layers of dilated residual convolutional blocks in the generator.
                Defaults to 30.
            gen_dilation_cycle (int): The number of layers with a given dilation before moving up by a power of two.
                `gen_n_layers` should be divisible by `dilation_cycle`. Defaults to 3.
            gen_dilated_kernel_size (int): Kernel size for the dilated conv1Ds in the generator. Defaults to 3.
            gen_residual_channels (int): Number of residual channels in the waveform generator. Defaults to 64.
            gen_skip_channels (int): Number of skip channels in the waveform generator. Defaults to 64.
            dis_out_channels (int): Number of output channels for the discriminator. Defaults to 1.
            dis_n_layers (int): Number of layers in the discriminator. Defaults to 10.
            dis_kernel_size (int): Kernel size for the 1D convolutions in the discriminator. Defaults to 3.
            dis_conv_channels (int): Number of conv channels for the discriminator. Defaults to 64.
            dis_conv_stride (int): Convolution stride for the 1D convs in the discriminator. Defaults to 1.
            dis_relu_alpha (float): Leaky ReLU alpha (or "negative slope") for the discriminator. Defaults to 0.2.
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
