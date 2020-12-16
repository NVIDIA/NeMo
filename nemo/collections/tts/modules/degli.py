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

# MIT License

# Copyright (c) 2019 Jeongmin Liu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.tts.helpers.helpers import OperationMode
from nemo.core.classes import Exportable, NeuralModule, typecheck
from nemo.core.neural_types.elements import IntType, LengthsType, SpectrogramType
from nemo.core.neural_types.neural_type import NeuralType


def str2act(txt):
    """Translates text to neural network activation"""
    return {
        "sigmoid": nn.Sigmoid(),
        "relu": nn.ReLU(),
        "none": nn.Sequential(),
        "lrelu": nn.LeakyReLU(0.1),
        "selu": nn.SELU(),
    }[txt.lower()]


def replace_magnitude(x, mag):
    """
    Extract the phase from x and apply it on mag
    x [B,2,F,T] : A tensor, where [:,0,:,:] is real and [:,1,:,:] is imaginary
    mag [B,1,F,T] : A tensor containing the absolute magnitude.
    """

    phase = torch.atan2(x[:, 1:], x[:, :1])  # imag, real
    return torch.cat([mag * torch.cos(phase), mag * torch.sin(phase)], dim=1)


def overlap_add(x, hop_length, eye=None):
    """
    component which is used for STFT
    x: B, W, T
    eye: identity matrix of size (W, W)
    return: B, W + hop_length * (T - 1)
    """
    n_batch, W, _ = x.shape
    if eye is None:
        eye = torch.eye(W, device=x.device)

    x = F.conv_transpose1d(x, eye, stride=hop_length, padding=0)
    x = x.view(n_batch, -1)
    return x


class InverseSTFT(nn.Module):
    """
    An STFT implementation for pytorch.
    """

    def __init__(self, n_fft, hop_length=None, win_length=None, window=None):
        super().__init__()

        self.n_fft = n_fft

        # Set the default hop if it's not already specified
        if hop_length is None:
            self.hop_length = int(win_length // 4)
        else:
            self.hop_length = hop_length

        if win_length is None:
            win_length = n_fft

        # kernel for overlap_add
        eye = torch.eye(n_fft)
        self.eye = nn.Parameter(eye.unsqueeze(1), requires_grad=False)  # n_fft, 1, n_fft

        # default window is a rectangular window (convention of torch.stft)
        if window is None:
            window = torch.ones(win_length)
        else:
            assert win_length == len(window)

        # pad window so that its length is n_fft
        diff = n_fft - win_length
        window = F.pad(window.unsqueeze(0), [diff // 2, math.ceil(diff / 2)])
        window.unsqueeze_(2)  # 1, n_fft, 1

        # square of window for calculating the numerical error occured during stft & istft
        self.win_sq = nn.Parameter(window ** 2, requires_grad=False)  # 1, n_fft, 1
        self.win_sq_sum = None

        # ifft basis * window
        # The reason why this basis is used instead of torch.ifft is
        # because torch.ifft / torch.irfft randomly cause segfault
        # when the model is in nn.DataParallel
        # of PyTorch 1.2.0 (py3.7_cuda10.0.130_cudnn7.6.2_01.2)
        eye_realimag = torch.stack((eye, torch.zeros(n_fft, n_fft)), dim=-1)
        basis = torch.fft.ifft(eye_realimag, signal_ndim=1)  # n_fft, n_fft, 2
        basis[..., 1] *= -1  # because (a+b*1j)*(c+d*1j) == a*c - b*d
        basis *= window
        self.basis = nn.Parameter(basis, requires_grad=False)  # n_fft, n_fft, 2

    def forward(self, stft_matrix, center=True, normalized=False, onesided=True, length=None):
        """stft_matrix: (n_batch (B), n_freq, n_frames (T), 2))
        if `onesided == True`, `n_freq == n_fft` should be satisfied.
        else, `n_freq == n_fft // 2+ 1` should be satisfied.

        """
        n_batch, n_freq, n_frames, _ = stft_matrix.shape

        assert (not onesided) and (n_freq == self.n_fft) or onesided and (n_freq == self.n_fft // 2 + 1)

        if length:
            padded_length = length
            if center:
                padded_length += self.n_fft
            n_frames = min(n_frames, math.ceil(padded_length / self.hop_length))

        stft_matrix = stft_matrix[:, :, :n_frames]

        if onesided:
            flipped = stft_matrix[:, 1:-1].flip(1)
            flipped[..., 1] *= -1
            stft_matrix = torch.cat((stft_matrix, flipped), dim=1)
            # now stft_matrix is (B, n_fft, T, 2)

        # The reason why this basis is used instead of torch.ifft is
        # because torch.ifft / torch.irfft randomly cause segfault
        # when the model is in nn.DataParallel
        # of PyTorch 1.2.0 (py3.7_cuda10.0.130_cudnn7.6.2_01.2)
        ytmp = torch.einsum('bftc,fwc->bwt', stft_matrix, self.basis)
        y = overlap_add(ytmp, self.hop_length, self.eye)
        # now y is (B, n_fft + hop_length * (n_frames - 1))

        # compensate numerical errors of window function
        if self.win_sq_sum is None or self.win_sq_sum.shape[1] != y.shape[1]:
            win_sq = self.win_sq.expand(1, -1, n_frames)  # 1, n_fft, n_frames
            win_sq_sum = overlap_add(win_sq, self.hop_length, self.eye)
            win_sq_sum[win_sq_sum <= torch.finfo(torch.float32).tiny] = 1.0
            # now win_sq_sum is (1, y.shape[1])
            self.win_sq_sum = win_sq_sum

        y /= self.win_sq_sum

        if center:
            y = y[:, self.n_fft // 2 :]
        if length is not None:
            if length < y.shape[1]:
                y = y[:, :length]
            else:
                y = F.pad(y, [0, length - y.shape[1]])
            # now y is (B, length)

        if normalized:
            y *= self.n_fft ** 0.5

        return y


class ConvGLU(nn.Module):
    """
    A convGlu operation, used by the Degli paper's model.
    """

    def __init__(self, in_ch, out_ch, kernel_size=(7, 7), padding=None, batchnorm=False, act="sigmoid", stride=None):
        super().__init__()
        if not padding:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        if stride is None:
            self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size, padding=padding)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size, padding=padding, stride=stride)
        self.weight = self.conv.weight
        self.bias = self.conv.bias

        if batchnorm:
            self.conv = nn.Sequential(self.conv, nn.BatchNorm2d(out_ch * 2))
        self.sigmoid = str2act(act)

    def forward(self, x):
        x = self.conv(x)
        ch = x.shape[1]
        x = x[:, : ch // 2, ...] * self.sigmoid(x[:, ch // 2 :, ...])
        return x


class DegliDNN(nn.Module):
    """
    The tiny model, which was used by the paper.
    Very efficient in terms of memory (~400KB),
    but utilizes 11x11 /7x7 convolutions which results slow execution.
    Inferece costs about the same at 5-32 ED model, with lower quality.
    """

    def __init__(self):
        super().__init__()

        k_x1, k_y1, k_x2, k_y2 = self.parse()
        ch_hidden = self.ch_hidden
        self.convglu_first = ConvGLU(6, ch_hidden, kernel_size=(k_y1, k_x1), batchnorm=True, act=self.act)
        self.two_convglus = nn.Sequential(
            ConvGLU(ch_hidden, ch_hidden, batchnorm=True, act=self.act, kernel_size=(k_y2, k_x2)),
            ConvGLU(ch_hidden, ch_hidden, act=self.act, kernel_size=(k_y2, k_x2)),
        )
        self.convglu_last = ConvGLU(ch_hidden, ch_hidden, act=self.act)

        self.conv = nn.Conv2d(ch_hidden, 2, kernel_size=(k_y2, k_x2), padding=((k_y2 - 1) // 2, (k_x2 - 1) // 2))

    def forward(self, x, mag_replaced, consistent, train_step=-1):
        x = torch.cat([x, mag_replaced, consistent], dim=1)
        x = self.convglu_first(x)
        residual = x
        x = self.two_convglus(x)
        x += residual
        x = self.convglu_last(x)
        x = self.conv(x)
        return x

    def parse(
        self, k_x1: int = 11, k_y1: int = 11, k_x2: int = 7, k_y2: int = 7, num_channel: int = 16, act="sigmoid"
    ):
        self.ch_hidden = num_channel

        self.act = act.lower()
        return (k_x1, k_y1, k_x2, k_y2)


class DegliED(nn.Module):

    """
    This is the default model.
    ED - Encoder-Decoder

    Arguments:
        layers: Number of encoder/deooders layers,
        k_x: conv kernels size in time domain,
        k_y: conv kernels size in frequency domain,
        s_x: conv stride in time domain ,
        s_y: conv stride in frequency domain domain,
        widening: a factor for expanding the number of channels during encoding phase,
        use_bn: use batch norm after encoders/decoders,
        linear_finalizer: apply fully connected after the final decoder.
        convGlu: use GLU after conv (recommended with sigmoid),
        act: string mapped by str2act() to the post encoder activation,
        act2: string mapped by str2act() to the post decoder activation,
        glu_bn: use batchnorm with the GLU operation, if enabled,
        use_weight_norm: use weigh-normalization,
    """

    def __init__(self, n_freq, config):
        super().__init__()

        self.parse(**config)

        layer_specs = [
            6,
            self.widening,
            self.widening * 2,
            self.widening * 4,
            self.widening * 8,
            self.widening * 8,
            self.widening * 8,
            self.widening * 8,
            self.widening * 8,
        ]

        layer_specs = layer_specs[0 : self.n_layers + 1]

        self.encoders = nn.ModuleList()

        conv, pad = self._gen_conv(
            layer_specs[0],
            layer_specs[1],
            convGlu=self.convGlu,
            rounding_needed=True,
            use_weight_norm=self.use_weight_norm,
        )

        # Encoder 0
        self.encoders.append(nn.Sequential(pad, conv))

        last_ch = layer_specs[1]
        self.lamb = 0
        # Rest of the encoders
        for i, ch_out in enumerate(layer_specs[2:]):
            d = OrderedDict()
            d['act'] = str2act(self.act)
            gain = math.sqrt(2.0 / (1.0 + self.lamb ** 2))
            gain = gain / math.sqrt(2)  # for naive signal propagation with residual w/o bn

            conv, pad = self._gen_conv(
                last_ch,
                ch_out,
                gain=gain,
                convGlu=self.convGlu,
                kernel_size=self.k_xy,
                use_weight_norm=self.use_weight_norm,
            )
            if not pad is None:
                d['pad'] = pad
            d['conv'] = conv

            if self.use_batchnorm:
                d['bn'] = nn.BatchNorm2d(ch_out)

            encoder_block = nn.Sequential(d)
            self.encoders.append(encoder_block)
            last_ch = ch_out

        layer_specs.reverse()
        self.decoders = nn.ModuleList()
        kernel_size = 4  # needed to match size of encoders inputs/outputs.
        # Decoders
        for i, ch_out in enumerate(layer_specs[1:]):

            d = OrderedDict()
            d['act'] = str2act(self.act2)
            gain = math.sqrt(2.0 / (1.0 + self.lamb ** 2))
            gain = gain / math.sqrt(2)

            if i == len(layer_specs) - 2:
                kernel_size = 5
                ch_out = 2
            conv = self._gen_deconv(last_ch, ch_out, gain=gain, k=kernel_size, use_weight_norm=self.use_weight_norm)
            d['conv'] = conv

            if self.use_batchnorm and i < self.n_layers - 1:
                d['bn'] = nn.BatchNorm2d(ch_out)

            decoder_block = nn.Sequential(d)
            self.decoders.append(decoder_block)
            last_ch = ch_out * 2

        if self.use_linear_finalizer:
            init_alpha = 0.001
            self.linear_finalizer = nn.Parameter(torch.ones(n_freq) * init_alpha, requires_grad=True)

    def parse(
        self,
        layers: int,
        k_x: int,
        k_y: int,
        s_x: int,
        s_y: int,
        widening: int,
        use_bn: bool,
        linear_finalizer: bool,
        convGlu: bool,
        act: str,
        act2: str,
        glu_bn: bool,
        use_weight_norm: bool,
    ):

        self.n_layers = layers
        self.k_xy = (k_y, k_x)
        self.s_xy = (s_y, s_x)
        self.widening = widening
        self.use_batchnorm = use_bn
        self.use_linear_finalizer = linear_finalizer
        self.convGlu = convGlu
        self.act = act
        self.act2 = act2
        self.glu_bn = glu_bn
        self.use_weight_norm = use_weight_norm

    def forward(self, x, mag_replaced, consistent, train_step=-1):
        x = torch.cat([x, mag_replaced, consistent], dim=1)

        encoders_output = []

        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            encoders_output.append(x)

        for i, decoder in enumerate(self.decoders[:-1]):
            x = decoder(x)
            x = torch.cat([x, encoders_output[-(i + 2)]], dim=1)

        x = self.decoders[-1](x)

        if self.use_linear_finalizer:
            x_perm = x.permute(0, 1, 3, 2)
            x = torch.mul(x_perm, self.linear_finalizer)
            x = x.permute(0, 1, 3, 2)

        return x

    def _gen_conv(
        self,
        in_ch,
        out_ch,
        strides=(2, 1),
        kernel_size=(5, 3),
        gain=math.sqrt(2),
        convGlu=False,
        rounding_needed=False,
        use_weight_norm=False,
    ):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        ky, kx = kernel_size
        p1x = (kx - 1) // 2
        p2x = kx - 1 - p1x
        p1y = (ky - 1) // 2
        p2y = ky - 1 - p1y

        if rounding_needed:
            pad_counts = (p1x, p2x, p1y - 1, p2y)
            pad = torch.nn.ReplicationPad2d(pad_counts)
        else:
            pad = None

        if convGlu:
            conv = ConvGLU(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=strides,
                batchnorm=self.glu_bn,
                padding=(0, 0),
                act="sigmoid",
            )
        else:
            if pad is None:
                conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=strides, padding=(p1y, p1x))
            else:
                conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=strides, padding=0)

        if use_weight_norm:
            conv = torch.nn.utils.weight_norm(conv, name='weight')

        w = conv.weight
        k = w.size(1) * w.size(2) * w.size(3)
        conv.weight.data.normal_(0.0, gain / math.sqrt(k))
        nn.init.constant_(conv.bias, 0.01)
        return conv, pad

    def _gen_deconv(self, in_ch, out_ch, strides=(2, 1), k=4, gain=math.sqrt(2), p=1, use_weight_norm=False):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]

        conv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=(k, 3), stride=strides, padding_mode='zeros', padding=(p, 1), dilation=1
        )

        if use_weight_norm:
            conv = torch.nn.utils.weight_norm(conv, name='weight')

        w = conv.weight
        k = w.size(1) * w.size(2) * w.size(3)
        conv.weight.data.normal_(0.0, gain / math.sqrt(k))
        nn.init.constant_(conv.bias, 0.01)

        return conv


class DegliModule(NeuralModule, Exportable):
    def __init__(self, n_fft: int, hop_length: int, depth: int, out_all_block: bool, tiny: bool, **kwargs):

        """
        Degli module

        Args:
            n_fft (int): STFT argument.
            hop_length (int): STFT argument.
            depth (int): depth > 1 will cause the NN to repeat, which new trainable weights.
            out_all_block (bool) output all blocks, not just the final output.
            tiny (bool) override all model hyperparameters and use the paper's small model.
            kwargs: to be passed for the dnn.
        """
        super().__init__()
        n_freq = n_fft // 2 + 1
        self.out_all_block = out_all_block
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        self.istft = InverseSTFT(n_fft, hop_length=hop_length, window=self.window.data)

        if tiny:
            self.dnns = nn.ModuleList([DegliDNN() for _ in range(depth)])
        else:
            self.dnns = nn.ModuleList([DegliED(n_freq, kwargs) for _ in range(depth)])

        self.mode = OperationMode.infer

    def stft(self, x):
        return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=False)

    @typecheck()
    def forward(self, x, mag, max_length=None, repeat=2):

        """

        Args:
            x: the tensor containing the phase [B,2,F,T]. Could be random for first iteration
            mag: the tensor containing the magnitude [B,1,F,T]
            max_length: maximum length of the audio in the batch
            repeat: how many iterations to apply using this network
        Returns:
            training/ validation:
                out_repeats: output of all blocks: [B, repeats, 2, F, T]
                final_out: output of the final block: [B, 2, F, T]
                residual: output of the final dnn: [B, 2, F, T]
            inference: the output of the final block
                final_out: output of the final block: [B, 2, F, T]
        """

        if self.training and self.mode != OperationMode.training:
            raise ValueError(f"{self} has self.training set to True but self.OperationMode was not set to training")
        if not self.training and self.mode == OperationMode.training:
            raise ValueError(f"{self} has self.training set to False but self.OperationMode was set to training")

        if isinstance(max_length, torch.Tensor):
            max_length = max_length.item()

        out_repeats = []
        for _ in range(repeat):
            for dnn in self.dnns:
                # B, 2, F, T
                mag_replaced = replace_magnitude(x, mag)

                # B, F, T, 2
                waves = self.istft(mag_replaced.permute(0, 2, 3, 1), length=max_length)
                consistent = self.stft(waves)

                # B, 2, F, T
                consistent = consistent.permute(0, 3, 1, 2)

                residual = dnn(x, mag_replaced, consistent)

                x = consistent - residual

            if self.out_all_block:
                out_repeats.append(x)

        final_out = replace_magnitude(x, mag)
        if self.mode == OperationMode.training or self.mode == OperationMode.validation:
            if self.out_all_block:
                out_repeats = torch.stack(out_repeats, dim=1)
            else:
                out_repeats = x.unsqueeze(1)

            return out_repeats, final_out, residual
        else:
            return final_out

    @property
    def input_types(self):
        return {
            "x": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "mag": NeuralType(('B', 'any', 'D', 'T'), SpectrogramType()),
            "max_length": NeuralType(None, LengthsType()),
            "repeat": NeuralType(None, IntType()),
        }

    @property
    def output_types(self):
        if self.mode == OperationMode.training or self.mode == OperationMode.validation:
            return {
                "out_repeats": NeuralType(('B', 'any', 'C', 'D', 'T'), SpectrogramType()),
                "final_out": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
                "residual": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            }
        else:
            return {
                "final_out": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            }

    def input_example(self):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        par = next(self.parameters())
        n_freq = self.n_fft // 2 + 1
        T = 500
        x = torch.randn((1, 2, n_freq, T), device=par.device, dtype=par.dtype)
        mag = torch.randn((1, 1, n_freq, T), device=par.device, dtype=par.dtype)
        max_length = torch.tensor((T - 1) * self.hop_length)
        repeat = torch.tensor(8)
        out = tuple([x, mag, max_length, repeat])
        return out
