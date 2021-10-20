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

import math

import torch
from torch import nn

from nemo.collections.tts.modules.aligner import ConvNorm
from nemo.collections.tts.modules.transformer import PositionalEmbedding


def average_f(f, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    f_cums = torch.nn.functional.pad(torch.cumsum(f, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_f = f.size(1)

    dcs = durs_cums_starts[:, None, :].expand(bs, n_f, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_f, l)

    f_sums = torch.gather(f_cums, 2, dce) - torch.gather(f_cums, 2, dcs)

    return f_sums / (durs.unsqueeze(1).float() + 1e-9)


def get_same_padding(kernel_size, stride, dilation) -> int:
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (dilation * (kernel_size - 1)) // 2


class SameLensMaskedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, groups,
                 use_conv_2d_with_last_channel_format=False):
        super().__init__()

        if not use_conv_2d_with_last_channel_format:
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                groups=groups,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_size),
                stride=(1, stride),
                dilation=(1, dilation),
                padding=(0, padding),
                groups=groups,
            ).to(memory_format=torch.channels_last)

        self.use_conv_2d_with_last_channel_format = use_conv_2d_with_last_channel_format

    def forward(self, x, mask):
        if not self.use_conv_2d_with_last_channel_format:
            x = self.conv(x.transpose(1, 2)).transpose(1, 2) * mask
        else:
            # x = x.transpose(1, 2).unsqueeze(-2) # .to(memory_format=torch.channels_last)
            # x = self.conv(x.transpose(1, 2).unsqueeze(-2)).squeeze(2).transpose(1, 2) * mask
            x = self.conv(x) * mask
        return x, mask


class SameLensMaskedLinear(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv_2d_with_last_channel_format):
        super().__init__()
        self.use_conv_2d_with_last_channel_format = use_conv_2d_with_last_channel_format
        if not use_conv_2d_with_last_channel_format:
            self.linear = nn.Linear(in_channels, out_channels)
        else:
            self.linear = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, mask):
        x = self.linear(x) * mask
        return x, mask


def create_time_mix_layer(in_feat, out_feat, use_conv_2d_with_last_channel_format):
    return SameLensMaskedLinear(in_feat, out_feat, use_conv_2d_with_last_channel_format)


def create_channel_mix_layer(in_feat, out_feat, kernel_size=3, stride=1, conv_type="depth-wise", dilation=1,
                             use_conv_2d_with_last_channel_format=False):
    padding = get_same_padding(kernel_size, stride=stride, dilation=dilation)

    if conv_type == "original":
        groups = 1
    elif conv_type == "depth-wise":
        groups = in_feat
    else:
        raise NotImplementedError

    conv = SameLensMaskedConv1d(
        in_feat, out_feat, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, groups=groups,
        use_conv_2d_with_last_channel_format=use_conv_2d_with_last_channel_format
    )

    return conv


class MLPBlock(nn.Module):
    def __init__(self, first_mix_layer, second_mix_layer, dropout):
        super().__init__()

        self.first_mix_layer = first_mix_layer
        self.act = nn.GELU()
        self.drop_1 = nn.Dropout(dropout)
        self.second_mix_layer = second_mix_layer
        self.drop_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x, mask = self.first_mix_layer(x, mask)
        x = self.act(x)
        x = self.drop_1(x)
        x, mask = self.second_mix_layer(x, mask)
        x = self.drop_2(x)
        return x, mask


class PreNormResidual(nn.Module):
    def __init__(self, fn, feature_dim, use_conv_2d_with_last_channel_format):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(feature_dim)
        self.use_conv_2d_with_last_channel_format = use_conv_2d_with_last_channel_format

    def forward(self, x, mask):
        if not self.use_conv_2d_with_last_channel_format:
            new_x, mask = self.fn(self.norm(x), mask)
        else:
            new_x, mask = self.fn(self.norm(x.transpose(1, 2)).transpose(1, 2), mask)

        x = x + new_x
        return x, mask


class MixerBlock(nn.Module):
    def __init__(self, in_feat, expansion_factor, kernel_size, conv_type, dropout, use_conv_2d_with_last_channel_format=False):
        super().__init__()

        self.channel_mix = PreNormResidual(
            fn=MLPBlock(
                first_mix_layer=create_channel_mix_layer(
                    in_feat=in_feat, out_feat=in_feat, kernel_size=kernel_size, conv_type=conv_type,
                    use_conv_2d_with_last_channel_format=use_conv_2d_with_last_channel_format
                ),
                second_mix_layer=create_channel_mix_layer(
                    in_feat=in_feat, out_feat=in_feat, kernel_size=kernel_size, conv_type=conv_type,
                    use_conv_2d_with_last_channel_format=use_conv_2d_with_last_channel_format
                ),
                dropout=dropout,
            ),
            feature_dim=in_feat,
            use_conv_2d_with_last_channel_format=use_conv_2d_with_last_channel_format
        )

        self.time_mix = PreNormResidual(
            fn=MLPBlock(
                first_mix_layer=create_time_mix_layer(
                    in_feat=in_feat, out_feat=expansion_factor * in_feat,
                    use_conv_2d_with_last_channel_format=use_conv_2d_with_last_channel_format
                ),
                second_mix_layer=create_time_mix_layer(
                    in_feat=expansion_factor * in_feat, out_feat=in_feat,
                    use_conv_2d_with_last_channel_format=use_conv_2d_with_last_channel_format
                ),
                dropout=dropout,
            ),
            feature_dim=in_feat,
            use_conv_2d_with_last_channel_format=use_conv_2d_with_last_channel_format
        )

    def forward(self, x, mask):
        x, mask = self.channel_mix(x, mask)
        x, mask = self.time_mix(x, mask)
        return x, mask


class TTSMixerModule(nn.Module):
    def __init__(
        self,
        num_tokens,
        feature_dim,
        num_layers,
        kernel_sizes,
        padding_idx=0,
        conv_type="depth-wise",
        expansion_factor=4,
        dropout=0.0,
        use_conv_2d_with_last_channel_format=False
    ):
        super().__init__()

        if len(kernel_sizes) != num_layers:
            raise ValueError

        self.d_model = feature_dim
        self.to_embed = (
            nn.Embedding(num_tokens, feature_dim, padding_idx=padding_idx) if num_tokens != -1 else nn.Identity()
        )
        self.use_conv_2d_with_last_channel_format = use_conv_2d_with_last_channel_format
        self.mixer_blocks = nn.Sequential(
            *[
                MixerBlock(feature_dim, expansion_factor, kernel_size, conv_type, dropout, use_conv_2d_with_last_channel_format)
                for kernel_size in kernel_sizes
            ],
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x, mask, conditioning=0):
        x = self.to_embed(x)
        x = x + conditioning

        if self.use_conv_2d_with_last_channel_format:
            x = x.transpose(1, 2).contiguous()
            mask = mask.transpose(1, 2).contiguous()

        x = x * mask
        for block in self.mixer_blocks:
            x, lens = block(x, mask)

        if not self.use_conv_2d_with_last_channel_format:
            x = self.norm(x)
        else:
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)

        if self.use_conv_2d_with_last_channel_format:
            x = x.transpose(1, 2).contiguous()
            mask = mask.transpose(1, 2).contiguous()

        return x, mask


class NLPAligner(nn.Module):
    """Module for alignment nlp tokens and text. """

    def __init__(self, n_text_channels=384, n_nlp_channels=128):
        super().__init__()

        self.text_pos_emb = PositionalEmbedding(n_text_channels)
        self.nlp_pos_emb = PositionalEmbedding(n_nlp_channels)

        self.query_proj = nn.Sequential(
            ConvNorm(n_text_channels, n_text_channels, kernel_size=3, bias=True, w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(n_text_channels, n_text_channels, kernel_size=1, bias=True),
        )

        self.key_proj = nn.Sequential(
            ConvNorm(n_nlp_channels, n_text_channels, kernel_size=3, bias=True, w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(n_text_channels, n_text_channels, kernel_size=1, bias=True),
        )

        self.value_proj = nn.Sequential(
            ConvNorm(n_nlp_channels, n_text_channels, kernel_size=3, bias=True, w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(n_text_channels, n_text_channels, kernel_size=1, bias=True),
        )

        self.scale = math.sqrt(n_text_channels)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Forward pass of the aligner encoder.

        Args:
            queries (torch.tensor): B x T1 x C1 tensor
            keys (torch.tensor): B x T2 x C2 tensor
            values (torch.tensor): B x T2 x C2 tensor
            q_mask (torch.tensor): B x T1 tensor, bool mask for variable length entries
            kv_mask (torch.tensor): B x T2 tensor, bool mask for variable length entries
        Output:
            attn_out (torch.tensor): B x T1 x C1 tensor
        """
        pos_q_seq = torch.arange(queries.size(-2), device=queries.device).to(queries.dtype)
        pos_kv_seq = torch.arange(keys.size(-2), device=queries.device).to(queries.dtype)

        pos_q_emb = self.text_pos_emb(pos_q_seq)
        pos_kv_emb = self.nlp_pos_emb(pos_kv_seq)

        if q_mask is not None:
            pos_q_emb = pos_q_emb * q_mask.unsqueeze(2)

        if kv_mask is not None:
            pos_kv_emb = pos_kv_emb * kv_mask.unsqueeze(2)

        queries = (queries + pos_q_emb).transpose(1, 2)
        keys = (keys + pos_kv_emb).transpose(1, 2)
        values = (values + pos_kv_emb).transpose(1, 2)

        queries_enc = self.query_proj(queries).transpose(-2, -1)  # B x T1 x C1
        keys_enc = self.key_proj(keys)  # B x C1 x T2
        values_enc = self.value_proj(values).transpose(-2, -1)  # B x T2 x C1

        scores = torch.matmul(queries_enc, keys_enc) / self.scale  # B x T1 x T2

        if kv_mask is not None:
            scores.masked_fill_(~kv_mask.unsqueeze(-2), -float("inf"))

        return torch.matmul(torch.softmax(scores, dim=-1), values_enc)  # B x T1 x C1
