# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from nemo.collections.asr.parts.utils.activations import Snake
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.collections.tts.parts.utils.helpers import mask_sequence_tensor
from nemo.core.classes.common import typecheck
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types.elements import AudioSignal, EncodedRepresentation, LengthsType, MelSpectrogramType, VoidType
from nemo.core.neural_types.neural_type import NeuralType


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def get_padding_2d(kernel_size: Tuple[int, int], dilation: Tuple[int, int]) -> Tuple[int, int]:
    paddings = (get_padding(kernel_size[0], dilation[0]), get_padding(kernel_size[1], dilation[1]))
    return paddings


def get_down_sample_padding(kernel_size: int, stride: int) -> int:
    return (kernel_size - stride + 1) // 2


def get_up_sample_padding(kernel_size: int, stride: int) -> Tuple[int, int]:
    output_padding = (kernel_size - stride) % 2
    padding = (kernel_size - stride + 1) // 2
    return padding, output_padding


class CodecActivation(nn.Module):
    """
    Choose between snake or Elu activation based on the input parameter.
    """

    def __init__(self, activation: str = "elu", channels: int = 1):
        super().__init__()
        activation = activation.lower()
        if activation == "snake":
            self.activation = Snake(channels)
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "lrelu":
            self.activation = torch.nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation {activation}")

    def forward(self, x):
        return self.activation(x)


class Conv1dNorm(NeuralModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: Optional[int] = None
    ):
        super().__init__()
        if not padding:
            padding = get_padding(kernel_size=kernel_size, dilation=dilation)

        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode="reflect",
        )
        self.conv = nn.utils.weight_norm(conv)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'T'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'C', 'T'), VoidType()),
        }

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    @typecheck()
    def forward(self, inputs, input_len):
        out = self.conv(inputs)
        out = mask_sequence_tensor(out, input_len)
        return out


class ConvTranspose1dNorm(NeuralModule):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding, output_padding = get_up_sample_padding(kernel_size, stride)
        conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            padding_mode="zeros",
        )
        self.conv = nn.utils.weight_norm(conv)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'T'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'C', 'T'), VoidType()),
        }

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    @typecheck()
    def forward(self, inputs, input_len):
        out = self.conv(inputs)
        out = mask_sequence_tensor(out, input_len)
        return out


class Conv2dNorm(NeuralModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1)
    ):
        super().__init__()
        assert len(kernel_size) == len(dilation)
        padding = get_padding_2d(kernel_size, dilation)
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode="reflect",
        )
        self.conv = nn.utils.weight_norm(conv)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'H', 'T'), VoidType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'C', 'H', 'T'), VoidType()),
        }

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    @typecheck()
    def forward(self, inputs):
        return self.conv(inputs)


class PeriodDiscriminator(NeuralModule):
    """
    Period discriminator introduced in HiFi-GAN https://arxiv.org/abs/2010.05646 which attempts to
    discriminate phase information by looking at equally spaced audio samples.

    Args:
        period: Spacing between audio sample inputs.
        lrelu_slope: Slope to use for activation. Leaky relu with slope of 0.1 or 0.2 is recommended for the
           stability of the feature matching loss.
    """

    def __init__(self, period, lrelu_slope=0.1):
        super().__init__()
        self.period = period
        self.activation = nn.LeakyReLU(lrelu_slope)
        self.conv_layers = nn.ModuleList(
            [
                Conv2dNorm(1, 32, kernel_size=(5, 1), stride=(3, 1)),
                Conv2dNorm(32, 128, kernel_size=(5, 1), stride=(3, 1)),
                Conv2dNorm(128, 512, kernel_size=(5, 1), stride=(3, 1)),
                Conv2dNorm(512, 1024, kernel_size=(5, 1), stride=(3, 1)),
                Conv2dNorm(1024, 1024, kernel_size=(5, 1), stride=(1, 1)),
            ]
        )
        self.conv_post = Conv2dNorm(1024, 1, kernel_size=(3, 1))

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "score": NeuralType(('B', 'C', 'T_out'), VoidType()),
            "fmap": [NeuralType(('B', 'D', 'T_layer', 'C'), VoidType())],
        }

    @typecheck()
    def forward(self, audio):

        batch_size, time = audio.shape
        out = rearrange(audio, 'B T -> B 1 T')
        # Pad audio so that it is divisible by the period
        if time % self.period != 0:
            n_pad = self.period - (time % self.period)
            out = F.pad(out, (0, n_pad), "reflect")
            time = time + n_pad
        # [batch, 1, (time / period), period]
        out = out.view(batch_size, 1, time // self.period, self.period)

        fmap = []
        for conv in self.conv_layers:
            # [batch, filters, (time / period / stride), period]
            out = conv(inputs=out)
            out = self.activation(out)
            fmap.append(out)
        # [batch, 1, (time / period / strides), period]
        score = self.conv_post(inputs=out)
        fmap.append(score)
        score = rearrange(score, "B 1 T C -> B C T")

        return score, fmap


class MultiPeriodDiscriminator(NeuralModule):
    """
    Wrapper class to aggregate results of multiple period discriminators.

    The periods are expected to be increasing prime numbers in order to maximize coverage and minimize overlap
    """

    def __init__(self, periods: Iterable[int] = (2, 3, 5, 7, 11), lrelu_slope=0.1):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(period=period, lrelu_slope=lrelu_slope) for period in periods]
        )

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T_audio'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "scores_real": [NeuralType(('B', 'C', 'T_out'), VoidType())],
            "scores_gen": [NeuralType(('B', 'C', 'T_out'), VoidType())],
            "fmaps_real": [[NeuralType(('B', 'D', 'T_layer', 'C'), VoidType())]],
            "fmaps_gen": [[NeuralType(('B', 'D', 'T_layer', 'C'), VoidType())]],
        }

    @typecheck()
    def forward(self, audio_real, audio_gen):
        scores_real = []
        scores_gen = []
        fmaps_real = []
        fmaps_gen = []
        for discriminator in self.discriminators:
            score_real, fmap_real = discriminator(audio=audio_real)
            score_gen, fmap_gen = discriminator(audio=audio_gen)
            scores_real.append(score_real)
            fmaps_real.append(fmap_real)
            scores_gen.append(score_gen)
            fmaps_gen.append(fmap_gen)

        return scores_real, scores_gen, fmaps_real, fmaps_gen


class Discriminator(NeuralModule):
    """
    Wrapper class which takes a list of discriminators and aggregates the results across them.
    """

    def __init__(self, discriminators: Iterable[NeuralModule]):
        super().__init__()
        self.discriminators = nn.ModuleList(discriminators)

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T_audio'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "scores_real": [NeuralType(('B', 'C', 'T_out'), VoidType())],
            "scores_gen": [NeuralType(('B', 'C', 'T_out'), VoidType())],
            "fmaps_real": [[NeuralType(('B', 'D', 'T_layer', 'C'), VoidType())]],
            "fmaps_gen": [[NeuralType(('B', 'D', 'T_layer', 'C'), VoidType())]],
        }

    @typecheck()
    def forward(self, audio_real, audio_gen):
        scores_real = []
        scores_gen = []
        fmaps_real = []
        fmaps_gen = []
        for discriminator in self.discriminators:
            score_real, score_gen, fmap_real, fmap_gen = discriminator(audio_real=audio_real, audio_gen=audio_gen)
            scores_real += score_real
            fmaps_real += fmap_real
            scores_gen += score_gen
            fmaps_gen += fmap_gen

        return scores_real, scores_gen, fmaps_real, fmaps_gen


class VectorQuantizerBase(NeuralModule, ABC):
    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "indices": NeuralType(('D', 'B', 'T'), Index()),
        }

    @typecheck()
    @abstractmethod
    def forward(self, inputs: torch.Tensor, input_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"indices": NeuralType(('D', 'B', 'T'), Index())},
    )
    @abstractmethod
    def encode(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        pass

    @typecheck(
        input_types={
            "indices": NeuralType(('D', 'B', 'T'), Index()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),},
    )
    @abstractmethod
    def decode(self, indices: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        pass


class FiniteScalarQuantizer(VectorQuantizerBase):
    """This quantizer is based on the Finite Scalar Quantization (FSQ) method.
    It quantizes each element of the input vector independently into a number of levels.

    Args:
        num_levels: number of levels for each dimension/element of the input vector
        eps: small regularization constant for scaling

    References:
        Mentzer et al., Finite Scalar Quantization: VQ-VAE Made Simple (https://arxiv.org/abs/2309.15505v1)
    """

    def __init__(self, num_levels: List[int], eps: float = 1e-3):
        super().__init__()
        self.period = period
        self.activation = torch.nn.LeakyReLU(0.1)
        self.conv_layers = nn.ModuleList(
            [
                Conv2dNorm(1, 32, kernel_size=(5, 1), stride=(3, 1)),
                Conv2dNorm(32, 128, kernel_size=(5, 1), stride=(3, 1)),
                Conv2dNorm(128, 512, kernel_size=(5, 1), stride=(3, 1)),
                Conv2dNorm(512, 1024, kernel_size=(5, 1), stride=(3, 1)),
                Conv2dNorm(1024, 1024, kernel_size=(5, 1), stride=(1, 1)),
            ]
        )
        self.conv_post = Conv2dNorm(1024, 1, kernel_size=(3, 1))

        # index base per dimension of the input vector
        # this is used to convert between per-dimension indices and a codebook token index
        dim_base_index = torch.cumprod(torch.tensor([1] + num_levels[:-1]), dim=0, dtype=torch.int32)
        dim_base_index = rearrange(dim_base_index, 'D -> 1 D 1')
        self.register_buffer('dim_base_index', dim_base_index)

        # Register the number of levels for each dimension
        num_levels = torch.tensor(num_levels, dtype=torch.int32)
        num_levels = rearrange(num_levels, 'D -> 1 D 1')
        self.register_buffer('num_levels', num_levels)

        # Regularization
        self.eps = eps

        logging.debug('Initializing %s with', self.__class__.__name__)
        logging.debug('\tdim:           %s', self.dim)
        logging.debug('\tnum_levels:    %s', self.num_levels)
        logging.debug('\tcodebook_size: %s', self.codebook_size)
        logging.debug('\teps:           %s', self.eps)

    @property
    def codebook_size(self):
        """Returns the size of the corresponding codebook."""
        return self.num_levels.prod().item()

    @property
    def dim(self):
        """Returns the dimension of the input vector."""
        return self.num_levels.numel()

    @property
    def codebook_dim(self):
        """Returns the dimension of the input vector.
        Keeping for compatiblitiy with the original RVQ implementation.
        """
        return self.dim

    @property
    def codes(self):
        """Returns the codebooks entries.

        Note that the codebook entries are implicitly defined by the number of levels.
        """
        indices = torch.arange(self.codebook_size)
        # [D, B, T]
        indices = rearrange(indices, 'B -> 1 B 1')
        # [B, D, T]
        codes = self.decode(indices=indices, input_len=None)
        # Remove the time dimension
        codes = codes.squeeze(-1)
        return codes

    @property
    def codebook(self):
        """Returns the codebooks entries.
        See self.codes for more details.
        """
        return self.codes

    @staticmethod
    def round(inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        """Round the input tensor to nearest integer
        and use a straight-through estimator for the gradient.
        """
        inputs_rounded = torch.round(inputs)
        return inputs + (inputs_rounded - inputs).detach()

    def compress(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        """Apply compression to the input, to limit to values.
        """
        output_scale = (self.num_levels - 1) / 2
        # scale down a bit to avoid rounding issues
        output_scale = output_scale * (1 - self.eps)
        # offset for even number of levels
        output_offset = torch.where(self.num_levels % 2 == 0, 0.5, 0)
        # shift for even number of levels
        input_shift = (output_offset / output_scale).tan()
        # compressed output
        output = output_scale * (inputs + input_shift).tanh() - output_offset
        return output

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"codes": NeuralType(('B', 'D', 'T'), Index())},
    )
    def inputs_to_codes(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        # apply compression
        compressed = self.compress(inputs=inputs, input_len=input_len)
        # apply rounding to nearest integer
        codes = self.round(inputs=compressed, input_len=input_len)
        # normalize to [-1, 1]
        scale = self.num_levels // 2
        codes = codes / scale
        return codes

    def codes_to_nonnegative(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert values centered arouund zero to nonnegative values.
        """
        scale = offset = self.num_levels // 2
        return scale * codes + offset

    def nonnegative_to_codes(self, codes_nonnegative: torch.Tensor) -> torch.Tensor:
        """Convert nonnegative values to values centered arouund zero.
        """
        scale = offset = self.num_levels // 2
        return (codes_nonnegative - offset) / scale

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Converts a code vector to a single index.
        """
        if codes.size(1) != self.dim:
            raise RuntimeError(
                f'Input code dimension {codes.size(1)} not matching the expected dimension {self.dim}, input codes shape {codes.shape}'
            )
        # convert code vectors to nonnegative values
        indices = self.codes_to_nonnegative(codes)
        # convert one nonnegative index per dimension to a single index per code vector
        indices = torch.sum(indices * self.dim_base_index, dim=1)
        return indices.to(torch.int32)

    # Implementation of VectorQuantiserBase API
    @typecheck()
    def forward(self, audio):
        # Pad audio
        batch_size, time = audio.shape
        out = rearrange(audio, 'B T -> B 1 T')
        if time % self.period != 0:
            n_pad = self.period - (time % self.period)
            out = F.pad(out, (0, n_pad), "reflect")
            time = time + n_pad
        out = out.view(batch_size, 1, time // self.period, self.period)

        fmap = []
        for conv in self.conv_layers:
            out = conv(inputs=out)
            out = self.activation(out)
            fmap.append(out)
        out = self.conv_post(inputs=out)
        fmap.append(out)
        out = rearrange(out, "B 1 T C -> B C T")

        return out, fmap


class GroupFiniteScalarQuantizer(VectorQuantizerBase):
    """Split the input vector into groups and apply FSQ on each group separately.
    This class is for convenience. Since FSQ is applied on each group separately,
    groups can be defined arbitrarily by splitting the input vector. However, this
    class makes it easy to construct several groups with the same quantization num_levels.

    Args:
        num_groups: number of groups to split the input into, each group will be quantized separately using num_codebooks//num_groups codebooks
        codebook_dim: embedding dimension, will be split into num_groups
        **kwargs: parameters of FiniteScalarQuantizer

    References:
        Yang et al, HiFi-Codec: Group-residual Vector quantization for High Fidelity Audio Codec, 2023 (http://arxiv.org/abs/2305.02765).
    """

    def __init__(self, num_groups: int, num_levels_per_group: List[int], **kwargs):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(period) for period in periods])

    @typecheck()
    def forward(self, inputs, input_len):
        skip_input = self.input_activation(inputs)
        skip_input = self.input_conv(inputs=skip_input, input_len=input_len)
        skip_input = self.skip_activation(skip_input)
        res = self.skip_conv(inputs=skip_input, input_len=input_len)
        res = self.dropout(res)
        out = inputs + res
        return out


class HiFiGANResBlock(NeuralModule):

    def __init__(self, channels, kernel_size, dilations, activation):
        super().__init__()

        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels=channels,
                filters=channels,
                kernel_size=kernel_size,
                dilation=dilation,
                activation=activation
            )
            for dilation in dilations
        ])

    def remove_weight_norm(self):
        for res_block in self.res_blocks:
            res_block.remove_weight_norm()

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'D', 'T'), VoidType())
        }

    @typecheck()
    def forward(self, inputs, input_len):
        out = inputs
        for res_block in self.res_blocks:
            out = res_block(inputs=out, input_len=input_len)
        return out


class HiFiGANResLayer(NeuralModule):

    def __init__(self, channels, kernel_sizes, dilations, activation):
        super().__init__()

        self.res_blocks = nn.ModuleList([
            HiFiGANResBlock(
                channels=channels,
                kernel_size=kernel_size,
                dilations=dilations,
                activation=activation
            )
            for kernel_size in kernel_sizes
        ])

    def remove_weight_norm(self):
        for res_block in self.res_blocks:
            res_block.remove_weight_norm()

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'D', 'T'), VoidType())
        }

    @typecheck()
    def forward(self, inputs, input_len):
        residuals = [res_block(inputs=inputs, input_len=input_len) for res_block in self.res_blocks]
        out = sum(residuals) / len(residuals)
        return out


class HiFiGANEncoder(NeuralModule):
    """
    Inverted version of HiFi-GAN generator
    """
    def __init__(
        self,
        down_sample_rates: Iterable[int] = (2, 4, 5, 8),
        base_channels: int = 32,
        in_kernel_size: int = 7,
        out_kernel_size: int = 7,
        encoded_dim: int = 128,
        resblock_kernel_sizes: Iterable[int] = (3, 7, 11),
        resblock_dilation_sizes: Iterable[int] = (1,),
        activation: str = "lrelu"
    ):
        assert in_kernel_size > 0
        assert out_kernel_size > 0

        super().__init__()

        self.down_sample_rates = down_sample_rates
        self.pre_conv = Conv1dNorm(in_channels=1, out_channels=base_channels, kernel_size=in_kernel_size)

        in_channels = base_channels
        self.activations = nn.ModuleList([])
        self.res_layers = nn.ModuleList([])
        self.down_sample_conv_layers = nn.ModuleList([])
        for i, down_sample_rate in enumerate(self.down_sample_rates):
            act = CodecActivation(activation, channels=in_channels)
            self.activations.append(act)

            res_layer = HiFiGANResLayer(
                channels=in_channels,
                kernel_sizes=resblock_kernel_sizes,
                dilations=resblock_dilation_sizes,
                activation=activation
            )
            self.res_layers.append(res_layer)


            out_channels = 2 * in_channels
            kernel_size = 2 * down_sample_rate
            down_sample_conv = Conv1dNorm(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=down_sample_rate,
                padding=get_down_sample_padding(kernel_size, down_sample_rate),
            )
            in_channels = out_channels
            self.down_sample_conv_layers.append(down_sample_conv)

        self.post_activation = CodecActivation(activation, channels=in_channels)
        self.post_conv = Conv1dNorm(in_channels=in_channels, out_channels=encoded_dim, kernel_size=out_kernel_size)

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "encoded": NeuralType(('B', 'D', 'T_encoded'), EncodedRepresentation()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
        }

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        self.post_conv.remove_weight_norm()
        for res_layer in self.res_layers:
            res_layer.remove_weight_norm()
        for down_sample_conv in self.down_sample_conv_layers:
            down_sample_conv.remove_weight_norm()

    @typecheck()
    def forward(self, audio, audio_len):
        encoded_len = audio_len
        audio = rearrange(audio, "B T -> B 1 T")
        # [B, C, T_audio]
        out = self.pre_conv(inputs=audio, input_len=encoded_len)
        for act, res_layer, down_sample_conv, down_sample_rate in zip(
            self.activations, self.res_layers, self.down_sample_conv_layers, self.down_sample_rates
        ):
            # [B, C, T]
            out = res_layer(inputs=out, input_len=encoded_len)
            out = act(out)

            encoded_len = encoded_len // down_sample_rate
            # [B, 2 * C, T / down_sample_rate]
            out = down_sample_conv(inputs=out, input_len=encoded_len)

        out = self.post_activation(out)
        # [B, encoded_dim, T_encoded]
        encoded = self.post_conv(inputs=out, input_len=encoded_len)
        return encoded, encoded_len


class HiFiGANDecoder(NeuralModule):
    def __init__(
        self,
        input_dim: int = 128,
        up_sample_rates: Iterable[int] = (8, 5, 4, 2),
        base_channels: int = 512,
        in_kernel_size: int = 7,
        out_kernel_size: int = 3,
        resblock_kernel_sizes: Iterable[int] = (3, 7, 11),
        resblock_dilation_sizes: Iterable[int] = (1, 3, 5),
        activation: Optional[str] = "lrelu"

    ):
        assert in_kernel_size > 0
        assert out_kernel_size > 0

        super().__init__()

        self.up_sample_rates = up_sample_rates
        self.pre_conv = Conv1dNorm(in_channels=input_dim, out_channels=base_channels, kernel_size=in_kernel_size)

        in_channels = base_channels
        self.activations = nn.ModuleList([])
        self.up_sample_conv_layers = nn.ModuleList([])
        self.res_layers = nn.ModuleList([])
        for i, up_sample_rate in enumerate(self.up_sample_rates):
            out_channels = in_channels // 2
            kernel_size = 2 * up_sample_rate

            act = CodecActivation(activation, channels=in_channels)
            self.activations.append(act)

            up_sample_conv = ConvTranspose1dNorm(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=up_sample_rate
            )
            in_channels = out_channels
            self.up_sample_conv_layers.append(up_sample_conv)

            res_layer = HiFiGANResLayer(
                channels=in_channels,
                kernel_sizes=resblock_kernel_sizes,
                dilations=resblock_dilation_sizes,
                activation=activation
            )
            self.res_layers.append(res_layer)

        self.post_activation = CodecActivation(activation, channels=in_channels)
        self.post_conv = Conv1dNorm(in_channels=in_channels, out_channels=1, kernel_size=out_kernel_size)
        self.out_activation = nn.Tanh()

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T_encoded'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        for up_sample_conv in self.up_sample_conv_layers:
            up_sample_conv.remove_weight_norm()
        for res_layer in self.res_layers:
            res_layer.remove_weight_norm()

    @typecheck()
    def forward(self, inputs, input_len):
        audio_len = input_len
        # [B, C, T_encoded]
        out = self.pre_conv(inputs=inputs, input_len=audio_len)
        for act, res_layer, up_sample_conv, up_sample_rate in zip(
            self.activations, self.res_layers, self.up_sample_conv_layers, self.up_sample_rates
        ):
            audio_len = audio_len * up_sample_rate
            out = act(out)
            # [B, C / 2, T * up_sample_rate]
            out = up_sample_conv(inputs=out, input_len=audio_len)
            out = res_layer(inputs=out, input_len=audio_len)

        out = self.post_activation(out)
        # [B, 1, T_audio]
        out = self.post_conv(inputs=out, input_len=audio_len)
        audio = self.out_activation(out)
        audio = rearrange(audio, "B 1 T -> B T")
        return audio, audio_len


class MelSpectrogramProcessor(NeuralModule):
    def __init__(self, mel_dim, sample_rate, win_length, hop_length, highfreq=None, log_guard=1.0):
        super(MelSpectrogramProcessor, self).__init__()
        self.hop_length = hop_length
        self.preprocessor = AudioToMelSpectrogramPreprocessor(
            sample_rate=sample_rate,
            highfreq=highfreq,
            features=mel_dim,
            pad_to=1,
            exact_pad=True,
            n_window_size=win_length,
            n_window_stride=hop_length,
            window_size=False,
            window_stride=False,
            n_fft=win_length,
            mag_power=1.0,
            log=True,
            log_zero_guard_type="add",
            log_zero_guard_value=log_guard,
            mel_norm=None,
            normalize=None,
            preemph=None,
            dither=0.0,
        )

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "spec": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
            "spec_len": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(self, audio, audio_len):
        # [batch, mel_dim, time]
        spec, spec_len = self.preprocessor(input_signal=audio, length=audio_len)
        return spec, spec_len


class ResNetEncoder(NeuralModule):
    def __init__(
        self,
        in_channels,
        out_channels=128,
        num_layers=6,
        hidden_channels=256,
        filters=768,
        kernel_size=3,
        dropout_rate=0.1,
        activation="lrelu"
    ):
        super(ResNetEncoder, self).__init__()

        self.pre_conv = Conv1dNorm(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size
        )
        self.res_layers = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=hidden_channels,
                    filters=filters,
                    kernel_size=kernel_size,
                    dropout_rate=dropout_rate,
                    activation=activation
                )
                for _ in range(num_layers)
            ]
        )
        self.post_activation = CodecActivation(activation, channels=hidden_channels)
        self.post_conv = Conv1dNorm(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size
        )

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        self.post_conv.remove_weight_norm()
        for res_layer in self.res_layers:
            res_layer.remove_weight_norm()

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "encoded": NeuralType(('B', 'C', 'T'), EncodedRepresentation())
        }

    @typecheck()
    def forward(self, inputs, input_len):
        encoded = self.pre_conv(inputs=inputs, input_len=input_len)
        for res_layer in self.res_layers:
            encoded = res_layer(inputs=encoded, input_len=input_len)
        encoded = self.post_activation(encoded)
        encoded = self.post_conv(inputs=encoded, input_len=input_len)
        return encoded


class SpeechEncoder(NeuralModule):
    def __init__(self, mel_processor, encoder):
        super(SpeechEncoder, self).__init__()
        self.mel_processor = mel_processor
        self.encoder = encoder

    def remove_weight_norm(self):
        self.encoder.remove_weight_norm()

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "encoded": NeuralType(('B', 'D', 'T_encoded'), EncodedRepresentation()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(self, audio, audio_len):
        out, spec_len = self.mel_processor(audio=audio, audio_len=audio_len)
        out = self.encoder(inputs=out, input_len=spec_len)
        return out, spec_len


class MultiBandMelEncoder(NeuralModule):
    def __init__(self, mel_bands, mel_processor, out_channels, num_layers=6, hidden_channels=128, filters=256):
        super(MultiBandMelEncoder, self).__init__()
        self.mel_bands = mel_bands
        self.mel_processor = mel_processor
        band_dims = [band[1] - band[0] for band in self.mel_bands]
        self.encoders = nn.ModuleList([
            ResNetEncoder(
                in_channels=band_dim,
                num_layers=num_layers,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                filters=filters
            )
            for band_dim in band_dims
        ])

    def remove_weight_norm(self):
        for encoder in self.encoders:
            encoder.remove_weight_norm()

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "encoded": NeuralType(('B', 'D', 'T_encoded'), EncodedRepresentation()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(self, audio, audio_len):
        spec, spec_len = self.mel_processor(audio=audio, audio_len=audio_len)
        outputs = []
        for (band_start, band_end), encoder in zip(self.mel_bands, self.encoders):
            # [batch_size, band_dim, time]
            spec_band = spec[:, band_start:band_end, :]
            # [batch_size, encoded_dim, time]
            band_out = encoder(inputs=spec_band, input_len=spec_len)
            outputs.append(band_out)
        # [batch_size, num_bands * encoded_dim, time]
        out = torch.cat(outputs, dim=1)
        return out, spec_len