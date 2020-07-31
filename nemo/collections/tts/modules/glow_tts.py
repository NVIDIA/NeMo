import math
import torch
from torch import nn

from nemo.collections.tts.modules import parts

from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import *
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils.decorators import experimental

@experimental
class DurationPredictor(NeuralModule):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = parts.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = parts.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    @property
    def input_types(self):
        return {
            "spect": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "mask": NeuralType(('B', 'D', 'T'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "durs": NeuralType(('B', 'T'), TokenLogDurationType()),
        }

    @typecheck()
    def forward(self, spect, mask):
        x = self.conv_1(spect * mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * mask)
        durs = x * mask
        return durs

    def save_to(self, save_path: str):
        """TODO: Implement"""
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """TODO: Implement"""
        pass


class TextEncoder(NeuralModule):
    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        filter_channels_dp,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        window_size=None,
        block_length=None,
        mean_only=False,
        prenet=False,
        gin_channels=0,
    ):

        super().__init__()

        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length
        self.mean_only = mean_only
        self.prenet = prenet
        self.gin_channels = gin_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        if prenet:
            self.pre = parts.ConvReluNorm(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                kernel_size=5,
                n_layers=3,
                p_dropout=0.5,
            )


        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for i in range(self.n_layers):
            self.attn_layers.append(
                parts.MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    window_size=window_size,
                    p_dropout=p_dropout,
                    block_length=block_length,
                )
            )
            self.norm_layers_1.append(parts.LayerNorm(hidden_channels))
            self.ffn_layers.append(
                parts.FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(parts.LayerNorm(hidden_channels))

        self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
        if not mean_only:
            self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj_w = DurationPredictor(
            hidden_channels + gin_channels, filter_channels_dp, kernel_size, p_dropout
        )

    @property
    def input_types(self):
        return {
            "text": NeuralType(('B', 'T'), TokenIndex()),
            "text_lengths": NeuralType(('B',), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "x_m": NeuralType(('B', 'D', 'T'), NormalDistributionMeanType()),
            "x_logs": NeuralType(('B', 'D', 'T'), NormalDistributionLogVarianceType()),
            "logw": NeuralType(('B', 'T'), TokenLogDurationType()),
            "x_mask": NeuralType(('B', 'D', 'T'), MaskType()),
        }

    @typecheck()
    def forward(self, text, text_lengths):


        x = self.emb(text) * math.sqrt(self.hidden_channels)  # [b, t, h]

        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(parts.sequence_mask(text_lengths, x.size(2)), 1).to(
            x.dtype
        )

        if self.prenet:
            x = self.pre(x, x_mask)

        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask

        x_m = self.proj_m(x) * x_mask
        if not self.mean_only:
            x_logs = self.proj_s(x) * x_mask
        else:
            x_logs = torch.zeros_like(x_m)

        logw = self.proj_w(spect=x.detach(), mask=x_mask)

        return x_m, x_logs, logw, x_mask

    def save_to(self, save_path: str):
        """TODO: Implement"""
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """TODO: Implement"""
        pass


class FlowSpecDecoder(NeuralModule):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_blocks,
        n_layers,
        p_dropout=0.0,
        n_split=4,
        n_sqz=2,
        sigmoid_scale=False,
        gin_channels=0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for b in range(n_blocks):
            self.flows.append(parts.ActNorm(channels=in_channels * n_sqz))
            self.flows.append(
                parts.InvConvNear(channels=in_channels * n_sqz, n_split=n_split)
            )
            self.flows.append(
                parts.CouplingBlock(
                    in_channels * n_sqz,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    gin_channels=gin_channels,
                    p_dropout=p_dropout,
                    sigmoid_scale=sigmoid_scale,
                )
            )

    @property
    def input_types(self):
        return {
            "spect": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "spect_mask": NeuralType(('B', 'D', 'T'), MaskType()),
            "reverse": NeuralType(elements_type=IntType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "z": NeuralType(('B', 'D', 'T'), NormalDistributionSamplesType()),
            "logdet_tot": NeuralType(('B',), LogDeterminantType()),
        }

    @typecheck()
    def forward(self, spect, spect_mask, reverse=False):
        if not reverse:
            flows = self.flows
            logdet_tot = 0
        else:
            flows = reversed(self.flows)
            logdet_tot = None

        x = spect
        x_mask = spect_mask

        if self.n_sqz > 1:
            x, x_mask = parts.squeeze(x, x_mask, self.n_sqz)
        for f in flows:
            if not reverse:
                x, logdet = f(x, x_mask, reverse=reverse)
                logdet_tot += logdet
            else:
                x, logdet = f(x, x_mask, reverse=reverse)
        if self.n_sqz > 1:
            x, x_mask = parts.unsqueeze(x, x_mask, self.n_sqz)
        return x, logdet_tot

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()

    def save_to(self, save_path: str):
        """TODO: Implement"""
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """TODO: Implement"""
        pass
