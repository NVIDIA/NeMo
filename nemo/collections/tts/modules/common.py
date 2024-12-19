# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

###############################################################################

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nemo.collections.tts.modules.submodules import ConvNorm, LinearNorm, MaskedInstanceNorm1d
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths, sort_tensor, unsort_tensor
from nemo.collections.tts.parts.utils.splines import (
    piecewise_linear_inverse_transform,
    piecewise_linear_transform,
    unbounded_piecewise_quadratic_transform,
)

import math
import torch

def _apply_sinc_resample_kernel(
    waveform: Tensor,
    orig_freq: int,
    new_freq: int,
    gcd: int,
    kernel: Tensor,
    width: int,
):
    if not waveform.is_floating_point():
        raise TypeError(f"Expected floating point type for waveform tensor, but received {waveform.dtype}.")

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    # pack batch
    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])

    num_wavs, length = waveform.shape
    waveform = torch.nn.functional.pad(waveform, (width, width + orig_freq))
    resampled = torch.nn.functional.conv1d(waveform[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
    target_length = torch.ceil(torch.as_tensor(new_freq * length / orig_freq)).long()
    resampled = resampled[..., :target_length]

    # unpack batch
    resampled = resampled.view(shape[:-1] + resampled.shape[-1:])
    return resampled

def _get_sinc_resample_kernel(
    orig_freq: int,
    new_freq: int,
    gcd: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: Optional[float] = None,
    device: torch.device = "cpu",
    dtype: Optional[torch.dtype] = None,
):
    if not (int(orig_freq) == orig_freq and int(new_freq) == new_freq):
        raise Exception(
            "Frequencies must be of integer type to ensure quality resampling computation. "
            "To work around this, manually convert both frequencies to integer values "
            "that maintain their resampling rate ratio before passing them into the function. "
            "Example: To downsample a 44100 hz waveform by a factor of 8, use "
            "`orig_freq=8` and `new_freq=1` instead of `orig_freq=44100` and `new_freq=5512.5`. "
            "For more information, please refer to https://github.com/pytorch/audio/issues/1487."
        )

    if resampling_method in ["sinc_interpolation", "kaiser_window"]:
        method_map = {
            "sinc_interpolation": "sinc_interp_hann",
            "kaiser_window": "sinc_interp_kaiser",
        }
        warnings.warn(
            f'"{resampling_method}" resampling method name is being deprecated and replaced by '
            f'"{method_map[resampling_method]}" in the next release. '
            "The default behavior remains unchanged.",
            stacklevel=3,
        )
    elif resampling_method not in ["sinc_interp_hann", "sinc_interp_kaiser"]:
        raise ValueError("Invalid resampling method: {}".format(resampling_method))

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    if lowpass_filter_width <= 0:
        raise ValueError("Low pass filter width should be positive.")
    base_freq = min(orig_freq, new_freq)
    # This will perform antialiasing filtering by removing the highest frequencies.
    # At first I thought I only needed this when downsampling, but when upsampling
    # you will get edge artifacts without this, as the edge is equivalent to zero padding,
    # which will add high freq artifacts.
    base_freq *= rolloff

    # The key idea of the algorithm is that x(t) can be exactly reconstructed from x[i] (tensor)
    # using the sinc interpolation formula:
    #   x(t) = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - t))
    # We can then sample the function x(t) with a different sample rate:
    #    y[j] = x(j / new_freq)
    # or,
    #    y[j] = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - j / new_freq))

    # We see here that y[j] is the convolution of x[i] with a specific filter, for which
    # we take an FIR approximation, stopping when we see at least `lowpass_filter_width` zeros crossing.
    # But y[j+1] is going to have a different set of weights and so on, until y[j + new_freq].
    # Indeed:
    # y[j + new_freq] = sum_i x[i] sinc(pi * orig_freq * ((i / orig_freq - (j + new_freq) / new_freq))
    #                 = sum_i x[i] sinc(pi * orig_freq * ((i - orig_freq) / orig_freq - j / new_freq))
    #                 = sum_i x[i + orig_freq] sinc(pi * orig_freq * (i / orig_freq - j / new_freq))
    # so y[j+new_freq] uses the same filter as y[j], but on a shifted version of x by `orig_freq`.
    # This will explain the F.conv1d after, with a stride of orig_freq.
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    # If orig_freq is still big after GCD reduction, most filters will be very unbalanced, i.e.,
    # they will have a lot of almost zero values to the left or to the right...
    # There is probably a way to evaluate those filters more efficiently, but this is kept for
    # future work.
    idx_dtype = dtype if dtype is not None else torch.float64

    idx = torch.arange(-width, width + orig_freq, dtype=idx_dtype, device=device)[None, None] / orig_freq

    t = torch.arange(0, -new_freq, -1, dtype=dtype, device=device)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)

    # we do not use built in torch windows here as we need to evaluate the window
    # at specific positions, not over a regular grid.
    if resampling_method == "sinc_interp_hann":
        window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    else:
        # sinc_interp_kaiser
        if beta is None:
            beta = 14.769656459379492
        beta_tensor = torch.tensor(float(beta))
        window = torch.i0(beta_tensor * torch.sqrt(1 - (t / lowpass_filter_width) ** 2)) / torch.i0(beta_tensor)

    t *= math.pi

    scale = base_freq / orig_freq
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels *= window * scale

    if dtype is None:
        kernels = kernels.to(dtype=torch.float32)

    return kernels, width


def torchaudio_functional_resample(
    waveform: Tensor,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: Optional[float] = None,
) -> Tensor:
    r"""Resamples the waveform at the new frequency using bandlimited interpolation. :cite:`RESAMPLE`.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Note:
        ``transforms.Resample`` precomputes and reuses the resampling kernel, so using it will result in
        more efficient computation if resampling multiple waveforms with the same resampling parameters.

    Args:
        waveform (Tensor): The input signal of dimension `(..., time)`
        orig_freq (int): The original frequency of the signal
        new_freq (int): The desired frequency
        lowpass_filter_width (int, optional): Controls the sharpness of the filter, more == sharper
            but less efficient. (Default: ``6``)
        rolloff (float, optional): The roll-off frequency of the filter, as a fraction of the Nyquist.
            Lower values reduce anti-aliasing, but also reduce some of the highest frequencies. (Default: ``0.99``)
        resampling_method (str, optional): The resampling method to use.
            Options: [``"sinc_interp_hann"``, ``"sinc_interp_kaiser"``] (Default: ``"sinc_interp_hann"``)
        beta (float or None, optional): The shape parameter used for kaiser window.

    Returns:
        Tensor: The waveform at the new frequency of dimension `(..., time).`
    """

    if orig_freq <= 0.0 or new_freq <= 0.0:
        raise ValueError("Original frequency and desired frequecy should be positive")

    if orig_freq == new_freq:
        return waveform

    gcd = math.gcd(int(orig_freq), int(new_freq))

    kernel, width = _get_sinc_resample_kernel(
        orig_freq,
        new_freq,
        gcd,
        lowpass_filter_width,
        rolloff,
        resampling_method,
        beta,
        waveform.device,
        waveform.dtype,
    )
    resampled = _apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd, kernel, width)
    return resampled


class TorchAudioResample(torch.nn.Module):
    r"""Resample a signal from one frequency to another. A resampling method can be given.
    Adapted from: https://github.com/pytorch/audio/blob/a6b0a140cc13216975e8922093459019537bb80a/src/torchaudio/transforms/_transforms.py#L898

    Args:
        orig_freq (int, optional): The original frequency of the signal. (Default: ``16000``)
        new_freq (int, optional): The desired frequency. (Default: ``16000``)
        resampling_method (str, optional): The resampling method to use.
            Options: [``sinc_interp_hann``, ``sinc_interp_kaiser``] (Default: ``"sinc_interp_hann"``)
        lowpass_filter_width (int, optional): Controls the sharpness of the filter, more == sharper
            but less efficient. (Default: ``6``)
        rolloff (float, optional): The roll-off frequency of the filter, as a fraction of the Nyquist.
            Lower values reduce anti-aliasing, but also reduce some of the highest frequencies. (Default: ``0.99``)
        beta (float or None, optional): The shape parameter used for kaiser window.
        dtype (torch.device, optional):
            Determnines the precision that resampling kernel is pre-computed and cached. If not provided,
            kernel is computed with ``torch.float64`` then cached as ``torch.float32``.
            If you need higher precision, provide ``torch.float64``, and the pre-computed kernel is computed and
            cached as ``torch.float64``. If you use resample with lower precision, then instead of providing this
            providing this argument, please use ``Resample.to(dtype)``, so that the kernel generation is still
            carried out on ``torch.float64``.

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.Resample(sample_rate, sample_rate/10)
        >>> waveform = transform(waveform)
    """

    def __init__(
        self,
        orig_freq: int = 16000,
        new_freq: int = 16000,
        resampling_method: str = "sinc_interp_hann",
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        beta: Optional[float] = None,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.gcd = math.gcd(int(self.orig_freq), int(self.new_freq))
        self.resampling_method = resampling_method
        self.lowpass_filter_width = lowpass_filter_width
        self.rolloff = rolloff
        self.beta = beta

        if self.orig_freq != self.new_freq:
            kernel, self.width = _get_sinc_resample_kernel(
                self.orig_freq,
                self.new_freq,
                self.gcd,
                self.lowpass_filter_width,
                self.rolloff,
                self.resampling_method,
                beta,
                dtype=dtype,
            )
            self.register_buffer("kernel", kernel)

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Output signal of dimension (..., time).
        """
        if self.orig_freq == self.new_freq:
            return waveform
        return _apply_sinc_resample_kernel(waveform, self.orig_freq, self.new_freq, self.gcd, self.kernel, self.width)


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b):
    t_act = torch.tanh(input_a)
    s_act = torch.sigmoid(input_b)
    acts = t_act * s_act
    return acts


class ExponentialClass(torch.nn.Module):
    def __init__(self):
        super(ExponentialClass, self).__init__()

    def forward(self, x):
        return torch.exp(x)


class DenseLayer(nn.Module):
    def __init__(self, in_dim=1024, sizes=[1024, 1024]):
        super(DenseLayer, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=True) for (in_size, out_size) in zip(in_sizes, sizes)]
        )

    def forward(self, x):
        for linear in self.layers:
            x = torch.tanh(linear(x))
        return x


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, lstm_norm_fn="spectral", max_batch_size=64):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        if lstm_norm_fn is not None:
            if 'spectral' in lstm_norm_fn:
                print("Applying spectral norm to LSTM")
                lstm_norm_fn_pntr = torch.nn.utils.spectral_norm
            elif 'weight' in lstm_norm_fn:
                print("Applying weight norm to LSTM")
                lstm_norm_fn_pntr = torch.nn.utils.weight_norm

        lstm_norm_fn_pntr(self.bilstm, 'weight_hh_l0')
        lstm_norm_fn_pntr(self.bilstm, 'weight_hh_l0_reverse')

        self.real_hidden_size: int = self.bilstm.proj_size if self.bilstm.proj_size > 0 else self.bilstm.hidden_size

        self.bilstm.flatten_parameters()

    def lstm_sorted(self, context: Tensor, lens: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        seq = nn.utils.rnn.pack_padded_sequence(context, lens.long().cpu(), batch_first=True, enforce_sorted=True)
        ret, _ = self.bilstm(seq, hx)
        return nn.utils.rnn.pad_packed_sequence(ret, batch_first=True)[0]

    def lstm(self, context: Tensor, lens: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        # To be ONNX-exportable, we need to sort here rather that while packing
        context, lens, unsort_ids = sort_tensor(context, lens)
        ret = self.lstm_sorted(context, lens, hx=hx)
        return unsort_tensor(ret, unsort_ids)

    def lstm_nocast(self, context: Tensor, lens: Tensor) -> Tensor:
        dtype = context.dtype
        # autocast guard is only needed for Torchscript to run in Triton
        # (https://github.com/pytorch/pytorch/issues/89241)
        with torch.amp.autocast(self.device.type, enabled=False):
            # Calculate sizes and prepare views to our zero buffer to pass as hx
            max_batch_size = context.shape[0]
            context = context.to(dtype=torch.float32)
            common_shape = (self.bilstm.num_layers * 2, max_batch_size)
            hx = (
                context.new_zeros(*common_shape, self.real_hidden_size),
                context.new_zeros(*common_shape, self.bilstm.hidden_size),
            )
            return self.lstm(context, lens, hx=hx).to(dtype=dtype)

    def forward(self, context: Tensor, lens: Tensor) -> Tensor:
        self.bilstm.flatten_parameters()
        if torch.jit.is_tracing():
            return self.lstm_nocast(context, lens)
        return self.lstm(context, lens)


class ConvLSTMLinear(nn.Module):
    def __init__(
        self,
        in_dim=None,
        out_dim=None,
        n_layers=2,
        n_channels=256,
        kernel_size=3,
        p_dropout=0.1,
        use_partial_padding=False,
        norm_fn=None,
    ):
        super(ConvLSTMLinear, self).__init__()
        self.bilstm = BiLSTM(n_channels, int(n_channels // 2), 1)
        self.convolutions = nn.ModuleList()

        if n_layers > 0:
            self.dropout = nn.Dropout(p=p_dropout)

        use_weight_norm = norm_fn is None

        for i in range(n_layers):
            conv_layer = ConvNorm(
                in_dim if i == 0 else n_channels,
                n_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1,
                w_init_gain='relu',
                use_weight_norm=use_weight_norm,
                use_partial_padding=use_partial_padding,
                norm_fn=norm_fn,
            )
            if norm_fn is not None:
                print("Applying {} norm to {}".format(norm_fn, conv_layer))
            else:
                print("Applying weight norm to {}".format(conv_layer))
            self.convolutions.append(conv_layer)

        self.dense = None
        if out_dim is not None:
            self.dense = nn.Linear(n_channels, out_dim)

    def forward(self, context: Tensor, lens: Tensor) -> Tensor:
        mask = get_mask_from_lengths(lens, context)
        mask = mask.to(dtype=context.dtype).unsqueeze(1)
        for conv in self.convolutions:
            context = self.dropout(F.relu(conv(context, mask)))
        # Apply Bidirectional LSTM
        context = self.bilstm(context.transpose(1, 2), lens=lens)
        if self.dense is not None:
            context = self.dense(context).permute(0, 2, 1)
        return context


def get_radtts_encoder(
    encoder_n_convolutions=3,
    encoder_embedding_dim=512,
    encoder_kernel_size=5,
    norm_fn=MaskedInstanceNorm1d,
):
    return ConvLSTMLinear(
        in_dim=encoder_embedding_dim,
        n_layers=encoder_n_convolutions,
        n_channels=encoder_embedding_dim,
        kernel_size=encoder_kernel_size,
        p_dropout=0.5,
        use_partial_padding=True,
        norm_fn=norm_fn,
    )


class Invertible1x1ConvLUS(torch.nn.Module):
    def __init__(self, c):
        super(Invertible1x1ConvLUS, self).__init__()
        # Sample a random orthonormal matrix to initialize weights
        W, _ = torch.linalg.qr(torch.FloatTensor(c, c).normal_())
        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        p, lower, upper = torch.lu_unpack(*torch.lu(W))

        self.register_buffer('p', p)
        # diagonals of lower will always be 1s anyway
        lower = torch.tril(lower, -1)
        lower_diag = torch.diag(torch.eye(c, c))
        self.register_buffer('lower_diag', lower_diag)
        self.lower = nn.Parameter(lower)
        self.upper_diag = nn.Parameter(torch.diag(upper))
        self.upper = nn.Parameter(torch.triu(upper, 1))

    @torch.amp.autocast(device_type='cuda', enabled=False)
    def forward(self, z, inverse=False):
        U = torch.triu(self.upper, 1) + torch.diag(self.upper_diag)
        L = torch.tril(self.lower, -1) + torch.diag(self.lower_diag)
        W = torch.mm(self.p, torch.mm(L, U))
        if inverse:
            if not hasattr(self, 'W_inverse'):
                # inverse computation
                W_inverse = W.float().inverse().to(dtype=z.dtype)
                self.W_inverse = W_inverse[..., None]
            z = F.conv1d(z, self.W_inverse.to(dtype=z.dtype), bias=None, stride=1, padding=0)
            return z
        else:
            W = W[..., None]
            z = F.conv1d(z, W, bias=None, stride=1, padding=0)
            log_det_W = torch.sum(torch.log(torch.abs(self.upper_diag)))
            return z, log_det_W


class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If inverse=True it does convolution with
    inverse
    """

    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0, bias=False)

        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z, inverse=False):
        # DO NOT apply n_of_groups, as it doesn't account for padded sequences
        W = self.conv.weight.squeeze()

        if inverse:
            if not hasattr(self, 'W_inverse'):
                # Inverse computation
                W_inverse = W.float().inverse().to(dtype=z.dtype)
                self.W_inverse = W_inverse[..., None]
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            # Forward computation
            log_det_W = torch.logdet(W).clone()
            z = self.conv(z)
            return z, log_det_W


class SimpleConvNet(torch.nn.Module):
    def __init__(
        self,
        n_mel_channels,
        n_context_dim,
        final_out_channels,
        n_layers=2,
        kernel_size=5,
        with_dilation=True,
        max_channels=1024,
        zero_init=True,
        use_partial_padding=True,
    ):
        super(SimpleConvNet, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.n_layers = n_layers
        in_channels = n_mel_channels + n_context_dim
        out_channels = -1
        self.use_partial_padding = use_partial_padding
        for i in range(n_layers):
            dilation = 2**i if with_dilation else 1
            padding = int((kernel_size * dilation - dilation) / 2)
            out_channels = min(max_channels, in_channels * 2)
            self.layers.append(
                ConvNorm(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                    w_init_gain='relu',
                    use_partial_padding=use_partial_padding,
                )
            )
            in_channels = out_channels

        self.last_layer = torch.nn.Conv1d(out_channels, final_out_channels, kernel_size=1)

        if zero_init:
            self.last_layer.weight.data *= 0
            self.last_layer.bias.data *= 0

    def forward(self, z_w_context, seq_lens: Optional[Tensor] = None):
        # seq_lens: tensor array of sequence sequence lengths
        # output should be b x n_mel_channels x z_w_context.shape(2)

        mask = get_mask_from_lengths(seq_lens, z_w_context).unsqueeze(1).to(dtype=z_w_context.dtype)

        for i in range(self.n_layers):
            z_w_context = self.layers[i](z_w_context, mask)
            z_w_context = torch.relu(z_w_context)

        z_w_context = self.last_layer(z_w_context)
        return z_w_context


class WN(torch.nn.Module):
    """
    Adapted from WN() module in WaveGlow with modififcations to variable names
    """

    def __init__(
        self,
        n_in_channels,
        n_context_dim,
        n_layers,
        n_channels,
        kernel_size=5,
        affine_activation='softplus',
        use_partial_padding=True,
    ):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        assert n_channels % 2 == 0
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        start = torch.nn.Conv1d(n_in_channels + n_context_dim, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start
        self.softplus = torch.nn.Softplus()
        self.affine_activation = affine_activation
        self.use_partial_padding = use_partial_padding
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2 * n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        for i in range(n_layers):
            dilation = 2**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = ConvNorm(
                n_channels,
                n_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                use_partial_padding=use_partial_padding,
                use_weight_norm=True,
            )
            self.in_layers.append(in_layer)
            res_skip_layer = nn.Conv1d(n_channels, n_channels, 1)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer)
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input: Tuple[Tensor, Tensor], seq_lens: Tensor = None):
        z, context = forward_input
        z = torch.cat((z, context), 1)  # append context to z as well
        z = self.start(z)
        output = torch.zeros_like(z)
        mask = None
        if self.use_partial_padding:
            mask = get_mask_from_lengths(seq_lens).unsqueeze(1).float()
        non_linearity = torch.relu
        if self.affine_activation == 'softplus':
            non_linearity = self.softplus

        for i in range(self.n_layers):
            z = non_linearity(self.in_layers[i](z, mask))
            res_skip_acts = non_linearity(self.res_skip_layers[i](z))
            output = output + res_skip_acts

        output = self.end(output)  # [B, dim, seq_len]
        return output


# Affine Coupling Layers
class SplineTransformationLayerAR(torch.nn.Module):
    def __init__(
        self,
        n_in_channels,
        n_context_dim,
        n_layers,
        affine_model='simple_conv',
        kernel_size=1,
        scaling_fn='exp',
        affine_activation='softplus',
        n_channels=1024,
        n_bins=8,
        left=-6,
        right=6,
        bottom=-6,
        top=6,
        use_quadratic=False,
    ):
        super(SplineTransformationLayerAR, self).__init__()
        self.n_in_channels = n_in_channels  # input dimensions
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.n_bins = n_bins
        self.spline_fn = piecewise_linear_transform
        self.inv_spline_fn = piecewise_linear_inverse_transform
        self.use_quadratic = use_quadratic

        if self.use_quadratic:
            self.spline_fn = unbounded_piecewise_quadratic_transform
            self.inv_spline_fn = unbounded_piecewise_quadratic_transform
            self.n_bins = 2 * self.n_bins + 1
        final_out_channels = self.n_in_channels * self.n_bins

        # autoregressive flow, kernel size 1 and no dilation
        self.param_predictor = SimpleConvNet(
            n_context_dim,
            0,
            final_out_channels,
            n_layers,
            with_dilation=False,
            kernel_size=1,
            zero_init=True,
            use_partial_padding=False,
        )

        # output is unnormalized bin weights

    def normalize(self, z, inverse):
        # normalize to [0, 1]
        if inverse:
            z = (z - self.bottom) / (self.top - self.bottom)
        else:
            z = (z - self.left) / (self.right - self.left)

        return z

    def denormalize(self, z, inverse):
        if inverse:
            z = z * (self.right - self.left) + self.left
        else:
            z = z * (self.top - self.bottom) + self.bottom

        return z

    def forward(self, z, context, inverse=False):
        b_s, c_s, t_s = z.size(0), z.size(1), z.size(2)

        z = self.normalize(z, inverse)

        if z.min() < 0.0 or z.max() > 1.0:
            print('spline z scaled beyond [0, 1]', z.min(), z.max())

        z_reshaped = z.permute(0, 2, 1).reshape(b_s * t_s, -1)
        affine_params = self.param_predictor(context)
        q_tilde = affine_params.permute(0, 2, 1).reshape(b_s * t_s, c_s, -1)
        with torch.amp.autocast(self.device.type, enabled=False):
            if self.use_quadratic:
                w = q_tilde[:, :, : self.n_bins // 2]
                v = q_tilde[:, :, self.n_bins // 2 :]
                z_tformed, log_s = self.spline_fn(z_reshaped.float(), w.float(), v.float(), inverse=inverse)
            else:
                z_tformed, log_s = self.spline_fn(z_reshaped.float(), q_tilde.float())

        z = z_tformed.reshape(b_s, t_s, -1).permute(0, 2, 1)
        z = self.denormalize(z, inverse)
        if inverse:
            return z

        log_s = log_s.reshape(b_s, t_s, -1)
        log_s = log_s.permute(0, 2, 1)
        log_s = log_s + c_s * (np.log(self.top - self.bottom) - np.log(self.right - self.left))
        return z, log_s


class SplineTransformationLayer(torch.nn.Module):
    def __init__(
        self,
        n_mel_channels,
        n_context_dim,
        n_layers,
        with_dilation=True,
        kernel_size=5,
        scaling_fn='exp',
        affine_activation='softplus',
        n_channels=1024,
        n_bins=8,
        left=-4,
        right=4,
        bottom=-4,
        top=4,
        use_quadratic=False,
    ):
        super(SplineTransformationLayer, self).__init__()
        self.n_mel_channels = n_mel_channels  # input dimensions
        self.half_mel_channels = int(n_mel_channels / 2)  # half, because we split
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.n_bins = n_bins
        self.spline_fn = piecewise_linear_transform
        self.inv_spline_fn = piecewise_linear_inverse_transform
        self.use_quadratic = use_quadratic

        if self.use_quadratic:
            self.spline_fn = unbounded_piecewise_quadratic_transform
            self.inv_spline_fn = unbounded_piecewise_quadratic_transform
            self.n_bins = 2 * self.n_bins + 1
        final_out_channels = self.half_mel_channels * self.n_bins

        self.param_predictor = SimpleConvNet(
            self.half_mel_channels,
            n_context_dim,
            final_out_channels,
            n_layers,
            with_dilation=with_dilation,
            kernel_size=kernel_size,
            zero_init=False,
        )

        # output is unnormalized bin weights

    def forward(self, z, context, inverse=False, seq_lens=None):
        b_s, c_s, t_s = z.size(0), z.size(1), z.size(2)

        # condition on z_0, transform z_1
        n_half = self.half_mel_channels
        z_0, z_1 = z[:, :n_half], z[:, n_half:]

        # normalize to [0,1]
        if inverse:
            z_1 = (z_1 - self.bottom) / (self.top - self.bottom)
        else:
            z_1 = (z_1 - self.left) / (self.right - self.left)

        z_w_context = torch.cat((z_0, context), 1)
        affine_params = self.param_predictor(z_w_context, seq_lens)
        z_1_reshaped = z_1.permute(0, 2, 1).reshape(b_s * t_s, -1)
        q_tilde = affine_params.permute(0, 2, 1).reshape(b_s * t_s, n_half, self.n_bins)

        with torch.amp.autocast(self.device.type, enabled=False):
            if self.use_quadratic:
                w = q_tilde[:, :, : self.n_bins // 2]
                v = q_tilde[:, :, self.n_bins // 2 :]
                z_1_tformed, log_s = self.spline_fn(z_1_reshaped.float(), w.float(), v.float(), inverse=inverse)
                if not inverse:
                    log_s = torch.sum(log_s, 1)
            else:
                if inverse:
                    z_1_tformed, _dc = self.inv_spline_fn(z_1_reshaped.float(), q_tilde.float(), False)
                else:
                    z_1_tformed, log_s = self.spline_fn(z_1_reshaped.float(), q_tilde.float())

        z_1 = z_1_tformed.reshape(b_s, t_s, -1).permute(0, 2, 1)

        # undo [0, 1] normalization
        if inverse:
            z_1 = z_1 * (self.right - self.left) + self.left
            z = torch.cat((z_0, z_1), dim=1)
            return z
        else:  # training
            z_1 = z_1 * (self.top - self.bottom) + self.bottom
            z = torch.cat((z_0, z_1), dim=1)
            log_s = log_s.reshape(b_s, t_s).unsqueeze(1) + n_half * (
                np.log(self.top - self.bottom) - np.log(self.right - self.left)
            )
            return z, log_s


class AffineTransformationLayer(torch.nn.Module):
    def __init__(
        self,
        n_mel_channels,
        n_context_dim,
        n_layers,
        affine_model='simple_conv',
        with_dilation=True,
        kernel_size=5,
        scaling_fn='exp',
        affine_activation='softplus',
        n_channels=1024,
        use_partial_padding=False,
    ):
        super(AffineTransformationLayer, self).__init__()
        if affine_model not in ("wavenet", "simple_conv"):
            raise Exception("{} affine model not supported".format(affine_model))
        if isinstance(scaling_fn, list):
            if not all([x in ("translate", "exp", "tanh", "sigmoid") for x in scaling_fn]):
                raise Exception("{} scaling fn not supported".format(scaling_fn))
        else:
            if scaling_fn not in ("translate", "exp", "tanh", "sigmoid"):
                raise Exception("{} scaling fn not supported".format(scaling_fn))

        self.affine_model = affine_model
        self.scaling_fn = scaling_fn
        if affine_model == 'wavenet':
            self.affine_param_predictor = WN(
                int(n_mel_channels / 2),
                n_context_dim,
                n_layers=n_layers,
                n_channels=n_channels,
                affine_activation=affine_activation,
                use_partial_padding=use_partial_padding,
            )
        elif affine_model == 'simple_conv':
            self.affine_param_predictor = SimpleConvNet(
                int(n_mel_channels / 2),
                n_context_dim,
                n_mel_channels,
                n_layers,
                with_dilation=with_dilation,
                kernel_size=kernel_size,
                use_partial_padding=use_partial_padding,
            )
        else:
            raise ValueError(
                f"Affine model is not supported: {affine_model}. Please choose either 'wavenet' or"
                f"'simple_conv' instead."
            )

        self.n_mel_channels = n_mel_channels

    def get_scaling_and_logs(self, scale_unconstrained):
        # (rvalle) re-write this
        if self.scaling_fn == 'translate':
            s = torch.exp(scale_unconstrained * 0)
            log_s = scale_unconstrained * 0
        elif self.scaling_fn == 'exp':
            s = torch.exp(scale_unconstrained)
            log_s = scale_unconstrained  # log(exp
        elif self.scaling_fn == 'tanh':
            s = torch.tanh(scale_unconstrained) + 1 + 1e-6
            log_s = torch.log(s)
        elif self.scaling_fn == 'sigmoid':
            s = torch.sigmoid(scale_unconstrained + 10) + 1e-6
            log_s = torch.log(s)
        elif isinstance(self.scaling_fn, list):
            s_list, log_s_list = [], []
            for i in range(scale_unconstrained.shape[1]):
                scaling_i = self.scaling_fn[i]
                if scaling_i == 'translate':
                    s_i = torch.exp(scale_unconstrained[:i] * 0)
                    log_s_i = scale_unconstrained[:, i] * 0
                elif scaling_i == 'exp':
                    s_i = torch.exp(scale_unconstrained[:, i])
                    log_s_i = scale_unconstrained[:, i]
                elif scaling_i == 'tanh':
                    s_i = torch.tanh(scale_unconstrained[:, i]) + 1 + 1e-6
                    log_s_i = torch.log(s_i)
                elif scaling_i == 'sigmoid':
                    s_i = torch.sigmoid(scale_unconstrained[:, i]) + 1e-6
                    log_s_i = torch.log(s_i)
                s_list.append(s_i[:, None])
                log_s_list.append(log_s_i[:, None])
            s = torch.cat(s_list, dim=1)
            log_s = torch.cat(log_s_list, dim=1)
        else:
            raise ValueError(
                f"Scaling function is not supported: {self.scaling_fn}. Please choose either 'translate', "
                f"'exp', 'tanh', or 'sigmoid' instead."
            )
        return s, log_s

    def forward(self, z, context, inverse=False, seq_lens=None):
        n_half = int(self.n_mel_channels / 2)
        z_0, z_1 = z[:, :n_half], z[:, n_half:]
        if self.affine_model == 'wavenet':
            affine_params = self.affine_param_predictor((z_0, context), seq_lens=seq_lens)
        elif self.affine_model == 'simple_conv':
            z_w_context = torch.cat((z_0, context), 1)
            affine_params = self.affine_param_predictor(z_w_context, seq_lens=seq_lens)
        else:
            raise ValueError(
                f"Affine model is not supported: {self.affine_model}. Please choose either 'wavenet' or "
                f"'simple_conv' instead."
            )

        scale_unconstrained = affine_params[:, :n_half, :]
        b = affine_params[:, n_half:, :]
        s, log_s = self.get_scaling_and_logs(scale_unconstrained)

        if inverse:
            z_1 = (z_1 - b) / s
            z = torch.cat((z_0, z_1), dim=1)
            return z
        else:
            z_1 = s * z_1 + b
            z = torch.cat((z_0, z_1), dim=1)
            return z, log_s


class ConvAttention(torch.nn.Module):
    def __init__(self, n_mel_channels=80, n_speaker_dim=128, n_text_channels=512, n_att_channels=80, temperature=1.0):
        super(ConvAttention, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.query_proj = Invertible1x1ConvLUS(n_mel_channels)

        self.key_proj = nn.Sequential(
            ConvNorm(n_text_channels, n_text_channels * 2, kernel_size=3, bias=True, w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(n_text_channels * 2, n_att_channels, kernel_size=1, bias=True),
        )

        self.query_proj = nn.Sequential(
            ConvNorm(n_mel_channels, n_mel_channels * 2, kernel_size=3, bias=True, w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(n_mel_channels * 2, n_mel_channels, kernel_size=1, bias=True),
            torch.nn.ReLU(),
            ConvNorm(n_mel_channels, n_att_channels, kernel_size=1, bias=True),
        )

    def forward(self, queries, keys, query_lens, mask=None, key_lens=None, attn_prior=None):
        """Attention mechanism for radtts. Unlike in Flowtron, we have no
        restrictions such as causality etc, since we only need this during
        training.

        Args:
            queries (torch.tensor): B x C x T1 tensor (likely mel data)
            keys (torch.tensor): B x C2 x T2 tensor (text data)
            query_lens: lengths for sorting the queries in descending order
            mask (torch.tensor): uint8 binary mask for variable length entries
                                 (should be in the T2 domain)
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask.
                                 Final dim T2 should sum to 1
        """
        temp = 0.0005
        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2
        # Beware can only do this since query_dim = attn_dim = n_mel_channels
        queries_enc = self.query_proj(queries)

        # Gaussian Isotopic Attention
        # B x n_attn_dims x T1 x T2
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2

        # compute log-likelihood from gaussian
        eps = 1e-8
        attn = -temp * attn.sum(1, keepdim=True)
        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + eps)

        attn_logprob = attn.clone()

        if mask is not None:
            attn.data.masked_fill_(mask.permute(0, 2, 1).unsqueeze(2), -float("inf"))

        attn = self.softmax(attn)  # softmax along T2
        return attn, attn_logprob


class GaussianDropout(torch.nn.Module):
    """
    Gaussian dropout using multiplicative gaussian noise.

    https://keras.io/api/layers/regularization_layers/gaussian_dropout/

    Can be an effective alternative bottleneck to VAE or VQ:

    https://www.deepmind.com/publications/gaussian-dropout-as-an-information-bottleneck-layer

    Unlike some other implementations, this takes the standard deviation of the noise as input
    instead of the 'rate' typically defined as: stdev = sqrt(rate / (1 - rate))
    """

    def __init__(self, stdev=1.0):
        super(GaussianDropout, self).__init__()
        self.stdev = stdev

    def forward(self, inputs):
        if not self.training:
            return inputs

        noise = torch.normal(mean=1.0, std=self.stdev, size=inputs.shape, device=inputs.device)
        out = noise * inputs
        return out
