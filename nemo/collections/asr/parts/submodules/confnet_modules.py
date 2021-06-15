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
#
import torch
from torch import nn as nn
from torch.nn import LayerNorm
from torch.nn import functional as F

from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
)
from nemo.collections.asr.parts.utils.activations import Swish

__all__ = ['ConfnetConvolution', 'ConfnetFeedForward', 'ConfnetLayer']


class ConfnetLayer(torch.nn.Module):
    """A single block of the Confnet encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        conv_kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
    """

    def __init__(
        self,
        d_model,
        d_ff,
        self_attention_model='rel_pos',
        n_heads=4,
        conv_kernel_size=31,
        dropout=0.1,
        dropout_att=0.1,
        pos_bias_u=None,
        pos_bias_v=None,
    ):
        super(ConfnetLayer, self).__init__()

        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.fc_factor = 0.5

        # first feed forward module
        self.norm_feed_forward1 = LayerNorm(d_model)
        self.feed_forward1 = ConfnetFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # convolution module
        self.norm_conv = LayerNorm(d_model)
        self.conv = ConfnetConvolution(d_model=d_model, kernel_size=conv_kernel_size)

        # multi-headed self-attention module
        self.norm_self_att = LayerNorm(d_model)
        # if self_attention_model == 'rel_pos':
        #     self.self_attn = RelPositionMultiHeadAttention(
        #         n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att, pos_bias_u=pos_bias_u, pos_bias_v=pos_bias_v
        #     )
        # elif self_attention_model == 'abs_pos':
        #     self.self_attn = MultiHeadAttention(n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att)
        # else:
        #     raise ValueError(
        #         f"'{self_attention_model}' is not not a valid value for 'self_attention_model', "
        #         f"valid values can be from ['rel_pos', 'abs_pos']"
        #     )
        self.fnet = FNetLayer(d_model=d_model, dropout=dropout)

        # second feed forward module
        self.norm_feed_forward2 = LayerNorm(d_model)
        self.feed_forward2 = ConfnetFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.norm_out = LayerNorm(d_model)

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
        Returns:
            x (torch.Tensor): (B, T, d_model)
        """
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        x = self.fc_factor * self.dropout(x) + residual

        residual = x
        x = self.norm_self_att(x)
        # if self.self_attention_model == 'rel_pos':
        #     x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb)
        # elif self.self_attention_model == 'abs_pos':
        #     x = self.self_attn(query=x, key=x, value=x, mask=att_mask)
        # else:
        #     x = None
        x = self.fnet(x)
        x = self.dropout(x) + residual

        residual = x
        x = self.norm_conv(x)
        x = self.conv(x, pad_mask)
        x = self.dropout(x) + residual

        residual = x
        x = self.norm_feed_forward2(x)
        x = self.feed_forward2(x)
        x = self.fc_factor * self.dropout(x) + residual

        x = self.norm_out(x)
        return x


class ConfnetConvolution(nn.Module):
    """The convolution module for the Confnet model.
    Args:
        d_model (int): hidden dimension
        kernel_size (int): kernel size for depthwise convolution
    """

    def __init__(self, d_model, kernel_size):
        super(ConfnetConvolution, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model * 2, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
            bias=True,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)

        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x, pad_mask=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)

        if pad_mask is not None:
            x.masked_fill_(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        return x


class ConfnetFeedForward(nn.Module):
    """
    feed-forward module of Confnet model.
    """

    def __init__(self, d_model, d_ff, dropout, activation=Swish()):
        super(ConfnetFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        num_hidden = expansion_factor * num_features
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        out = self.dropout2(self.fc2(x))
        return out

def fourier_transform(x):
    return torch.fft.fft2(x, dim=(-1, -2)).real


class FNetEncoderLayer(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        self.ff = FeedForward(d_model, expansion_factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = fourier_transform(x)
        x = self.norm1(x + residual)
        residual = x
        x = self.ff(x)
        out = self.norm2(x + residual)
        return out

class FNetLayer(nn.TransformerEncoder):
    def __init__(
        self, d_model=256, expansion_factor=2, dropout=0.1, num_layers=1,
    ):
        encoder_layer = FNetEncoderLayer(d_model, expansion_factor, dropout)
        super().__init__(encoder_layer=encoder_layer, num_layers=num_layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x