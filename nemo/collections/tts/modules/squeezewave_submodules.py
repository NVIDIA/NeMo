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
#
# Copyright (c) 2020 Tianren Gao, Bohan Zhai, Flora Xue,
# Daniel Rothchild, Bichen Wu, Joseph E. Gonzalez, Kurt Keutzer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch

from nemo.collections.tts.helpers.helpers import remove
from nemo.collections.tts.modules.submodules import fused_add_tanh_sigmoid_multiply


def fuse_conv_and_bn(conv, bn):
    fusedconv = torch.nn.Conv1d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        padding=conv.padding,
        bias=True,
        groups=conv.groups,
    )
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    w_bn = w_bn.clone()
    fusedconv.weight.data = torch.mm(w_bn, w_conv).view(fusedconv.weight.size())
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.weight.size(0))
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    b_bn = torch.unsqueeze(b_bn, 1)
    bn_3 = b_bn.expand(-1, 3)
    b = torch.matmul(w_conv, torch.transpose(bn_3, 0, 1))[range(b_bn.size()[0]), range(b_bn.size()[0])]
    fusedconv.bias.data = b_conv + b
    return fusedconv


def remove_batchnorm(conv_list):
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        depthwise = fuse_conv_and_bn(old_conv[1], old_conv[0])
        pointwise = old_conv[2]
        new_conv_list.append(torch.nn.Sequential(depthwise, pointwise))
    return new_conv_list


def remove_weightnorm(model):
    squeezewave = model
    for wavenet in squeezewave.wavenet:
        wavenet.start = torch.nn.utils.remove_weight_norm(wavenet.start)
        wavenet.in_layers = remove_batchnorm(wavenet.in_layers)
        wavenet.cond_layer = torch.nn.utils.remove_weight_norm(wavenet.cond_layer)
        wavenet.res_skip_layers = remove(wavenet.res_skip_layers)
    return squeezewave


class SqueezeWaveNet(torch.nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary
    difference from WaveNet is the convolutions need not be causal.  There is
    also no dilation size reset.  The dilation only doubles on each layer
    """

    def __init__(self, n_in_channels, n_mel_channels, n_layers, n_channels, kernel_size):
        super().__init__()
        assert kernel_size % 2 == 1
        assert n_channels % 2 == 0
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

        start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2 * n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(n_mel_channels, 2 * n_channels * n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        padding = (kernel_size - 1) // 2
        for _ in range(n_layers):
            self.in_layers.append(
                torch.nn.Sequential(
                    torch.nn.BatchNorm1d(n_channels),
                    torch.nn.Conv1d(n_channels, n_channels, kernel_size, padding=padding, groups=n_channels),
                    torch.nn.Conv1d(n_channels, 2 * n_channels, 1),
                )
            )

            res_skip_layer = torch.nn.Conv1d(n_channels, n_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input):
        audio, spect = forward_input
        audio = self.start(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        for i in range(self.n_layers):
            spect_offset = i * 2 * self.n_channels
            cond = spect[:, spect_offset : spect_offset + 2 * self.n_channels, :]
            if cond.size(2) < audio.size(2):
                cond = self.upsample(cond)

            acts = fused_add_tanh_sigmoid_multiply(self.in_layers[i](audio), cond, n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            audio = audio + res_skip_acts

        return self.end(audio)
