# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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


import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.tts.helpers.helpers import get_mask_from_lengths

from nemo.collections.tts.modules.transformer import PositionalEmbedding, TransformerLayer
from nemo.utils import logging


class FFTransformer(nn.Module):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_head,
        d_inner,
        kernel_size,
        dropout,
        dropatt,
        dropemb=0.0,
        pre_lnorm=False,
        embed_input=True,
        n_embed=84,
        padding_idx=83,
    ):
        super(FFTransformer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.padding_idx = padding_idx

        if embed_input:
            self.word_emb = nn.Embedding(n_embed, d_model, padding_idx=self.padding_idx)
        else:
            self.word_emb = None

        self.pos_emb = PositionalEmbedding(self.d_model)
        self.drop = nn.Dropout(dropemb)
        self.layers = nn.ModuleList()

        for _ in range(n_layer):
            self.layers.append(
                TransformerLayer(
                    n_head, d_model, d_head, d_inner, kernel_size, dropout, dropatt=dropatt, pre_lnorm=pre_lnorm
                )
            )

    def forward(self, dec_inp, seq_lens=None):
        if self.word_emb is None:
            inp = dec_inp
            mask = get_mask_from_lengths(seq_lens).unsqueeze(2)
        else:
            inp = self.word_emb(dec_inp)
            # [bsz x L x 1]
            mask = (dec_inp != self.padding_idx).unsqueeze(2)

        pos_seq = torch.arange(inp.size(1), device=inp.device, dtype=inp.dtype)
        pos_emb = self.pos_emb(pos_seq) * mask
        out = self.drop(inp + pos_emb)

        for layer in self.layers:
            out = layer(out, mask=mask)

        # out = self.drop(out)
        return out, mask


class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


class VariancePredictor(nn.Module):
    def __init__(self, d_model, d_inner, kernel_size, dropout):
        """
        Variance predictor submodule for FastSpeech 2/2s, used for pitch and energy prediction.

        Args:
            d_model: Input dimension.
            d_inner: Hidden dimension of the variance predictor.
            kernel_size: Conv1d kernel size.
            dropout: Dropout value for the variance predictor.
        """
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.kernel_size = kernel_size

        self.layers = nn.Sequential(
            Transpose(),
            nn.Conv1d(d_model, d_inner, kernel_size, stride=1, padding=(kernel_size // 2)),
            nn.ReLU(),
            Transpose(),
            nn.LayerNorm(d_inner),
            Transpose(),
            nn.Dropout(dropout),
            nn.Conv1d(d_inner, d_inner, kernel_size, stride=1, padding=(kernel_size // 2)),
            nn.ReLU(),
            Transpose(),
            nn.LayerNorm(d_inner),
            nn.Dropout(dropout),
            nn.Linear(d_inner, 1),
        )

    def forward(self, vp_input):
        return self.layers(vp_input).squeeze(-1)


class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hiddens, durations):
        """
        Expands the hidden states according to the duration target/prediction (depends on train vs inference).

        Args:
            hiddens: Hidden states of dimension (batch, time, emb_dim)
            durations: Timings for each frame of the hiddens, dimension (batch, time)
        """
        # Find max expanded length over batch elements for padding
        max_len = torch.max(torch.sum(durations, 1))

        out_list = []
        for x, d in zip(hiddens, durations):
            # For frame i of a single batch element x, repeats each the frame d[i] times.
            repeated = torch.cat([x[i].repeat(d[i], 1) for i in range(d.numel()) if d[i] != 0])
            repeated = F.pad(repeated, (0, 0, 0, max_len - repeated.shape[0]), "constant", value=0.0)
            out_list.append(repeated)

        return torch.stack(out_list)


class LengthRegulator2(nn.Module):
    def forward(self, x, dur):
        output = []
        for x_i, dur_i in zip(x, dur):
            expanded = self.expand(x_i, dur_i)
            output.append(expanded)
        output = self.pad(output)
        return output

    def expand(self, x, dur):
        output = []
        for i, frame in enumerate(x):
            expanded_len = int(dur[i] + 0.5)
            expanded = frame.expand(expanded_len, -1)
            output.append(expanded)
        output = torch.cat(output, 0)
        return output

    def pad(self, x):
        output = []
        max_len = max([x[i].size(0) for i in range(len(x))])
        for i, seq in enumerate(x):
            padded = F.pad(seq, [0, 0, 0, max_len - seq.size(0)], 'constant', 0.0)
            output.append(padded)
        output = torch.stack(output)
        return output


class DilatedResidualConvBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, dilation, kernel_size):
        """
        Dilated residual convolutional block for the waveform decoder.
        residual_channels = input dimension. Input: (batch, residual_channels, time)
        """
        super().__init__()

        self.n_channels = residual_channels

        # Dilated conv
        padding = int((kernel_size * dilation - dilation) / 2)
        self.dilated_conv = nn.Conv1d(
            in_channels=self.n_channels,
            out_channels=(2 * self.n_channels),
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        # Pointwise conv for residual
        self.pointwise_conv_residual = nn.Conv1d(
            in_channels=self.n_channels, out_channels=residual_channels, kernel_size=1
        )

        # Pointwise conv for skip connection (this is separate from resids but not mentioned in the WaveNet paper)
        self.pointwise_conv_skip = nn.Conv1d(in_channels=self.n_channels, out_channels=skip_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.dilated_conv(x)
        out = nn.Tanh(out[:, : self.n_channels, :]) * torch.sigmoid(out[:, self.n_channels :, :])

        # Skip connection
        skip_out = self.pointwise_conv_skip(out)

        # Residual connection
        out = (out + residual) * torch.sqrt(0.5)

        return skip_out, out


def _conv_weight_norm(module):
    """
    Function to apply weight norm to only convolutional layers in the waveform decoder.
    """
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d):
        nn.utils.weight_norm(module)


class WaveformGenerator(nn.Module):
    def __init__(
        self,
        in_channels=256,
        out_channels=1,
        trans_kernel_size=64,
        hop_size=256,
        n_layers=30,
        dilation_cycle=3,
        dilated_kernel_size=3,
        residual_channels=64,
        skip_channels=64,
    ):
        """
        Waveform generator for FastSpeech 2s, based on WaveNet and Parallel WaveGAN.
        """
        if n_layers // dilation_cycle != 0:
            logging.error(
                f"Number of layers in dilated residual convolution blocks should be divisible by dilation cycle."
                f" Have {n_layers} layers and cycle size {dilation_cycle}, which are not divisible."
            )

        self.n_layers = n_layers

        # Transposed 1D convolution to upsample slices of hidden reps to a longer audio length
        # TODO: double-check transposed conv args. -- kernel size in particular.
        #       The FastSpeech 2 paper says "filter size 64," Huihan's repo uses kernel_size=3.
        self.transposed_conv = nn.ConvTranspose1d(
            in_channels=in_channels, out_channels=residual_channels, kernel_size=trans_kernel_size, stride=hop_size,
        )

        # Repeated dilated residual convolution blocks
        self.dilated_res_conv_blocks = nn.ModuleList()
        dilation = 1

        for i in range(n_layers):
            self.dilated_res_conv_blocks.append(
                DilatedResidualConvBlock(
                    residual_channels=residual_channels,
                    skip_channels=skip_channels,
                    dilation=dilation,
                    kernel_size=dilated_kernel_size,
                )
            )
            # Increase dilation by a factor of 2 every {dilation_cycle}-layers.
            if (i + 1) % dilation_cycle == 0:
                dilation *= 2

        # Output activations and pointwise convolutions
        self.out_layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, out_channels, kernel_size=1),
        )

        # Apply weight norm to conv layers
        self.apply(_conv_weight_norm)

    def forward(self, x, use_softmax=False):
        # Expand via upsampling
        x = self.transposed_conv(x)

        # Dilated conv blocks
        skip_outs = 0
        for i in range(self.n_layers):
            skip_out, x = self.dilated_res_conv_blocks[i](x)
            skip_outs += skip_out
        skip_outs *= torch.sqrt(1.0 / self.n_layers)

        # Output layers
        out = self.out_layers(skip_outs)

        if use_softmax:
            out = nn.Softmax(out, dim=1)

        return out


class WaveformDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        n_layers=10,
        kernel_size=3,
        conv_channels=64,
        conv_stride=1,
        relu_alpha=0.2,
    ):
        """
       Waveform discriminator for FastSpeech 2s, based on Parallel WaveGAN.
       """
        # Layers of non-causal dilated 1D convolutions and leaky ReLU
        self.layers = nn.ModuleList()
        prev_channels = in_channels
        channels = conv_channels

        for i in range(n_layers - 1):
            # Dilated 1D conv
            dilation = i if i > 0 else 1
            padding = int((kernel_size * dilation - dilation) / 2)
            self.layers.append(
                nn.Conv1d(
                    in_channels=prev_channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    stride=conv_stride,
                )
            )
            prev_channels = channels

            # Leaky ReLU
            self.layers.append(nn.LeakyReLU(negative_slope=relu_alpha, inplace=True))

        # Last layer
        self.layer.append(
            nn.Conv1d(
                in_channels=prev_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=int((kernel_size - 1) / 2),
            )
        )

        # Apply weight norm to conv layers
        self.apply(_conv_weight_norm)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
