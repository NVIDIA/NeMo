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

import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.modules import LayerNorm

from nemo.collections.asr.modules.conformer_modules import ConformerConvolution, ConformerFeedForward
from nemo.collections.asr.modules.multi_head_attention import (
    MultiHeadAttention,
    PositionalEncoding,
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
)
from nemo.collections.asr.modules.subsampling import ConvSubsampling
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType, SpectrogramType

__all__ = ['ConformerEncoder']


class ConformerEncoder(NeuralModule, Exportable):
    """
    The encoder for ASR model of Conformer.
    Based on this paper:
    'Conformer: Convolution-augmented Transformer for Speech Recognition' by Anmol Gulati et al.
    https://arxiv.org/abs/2005.08100

    Args:
        feat_in (int): the size of feature channels
        n_layers (int): number of layers of ConformerBlock
        d_model (int): the hidden size of the model
        feat_out (int): the size of the output features
            Defaults to -1 (means feat_out is d_model)
        subsampling (str): the method of subsampling, choices=['vggnet', 'striding']
        subsampling_factor (int): the subsampling factor which should be power of 2
            Defaults to 4.
        subsampling_conv_channels (int): the size of the convolutions in the subsampling module
            Defaults to 64.
        ff_expansion_factor (int): the expansion factor in feed forward layers
            Defaults to 4.
        self_attention_model (str): type of the attention layer and positional encoding
            choices=['rel_pos', 'abs_pos'].
        n_heads (int): number of heads in multi-headed attention layers
            Defaults to 4.
        xscaling (bool): enables scaling the inputs to the multi-headed attention layers by sqrt(d_model)
            Defaults to True.
        conv_kernel_size (int): the size of the convolutions in the convolutional modules
            Defaults to 31.
        dropout (float): the dropout rate used in all layers except the attention layers
            Defaults to 0.1.
        dropout_emb (float): the dropout rate used for the positional embeddings
            Defaults to 0.1.
        dropout_att (float): the dropout rate used for the attention layer
            Defaults to 0.0.
    """

    def _prepare_for_export(self):
        Exportable._prepare_for_export(self)

    def input_example(self):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        input_example = torch.randn(16, self.__feat_in, 256).to(next(self.parameters()).device)
        return tuple([input_example])

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            }
        )

    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        feat_out=-1,
        subsampling='vggnet',
        subsampling_factor=4,
        subsampling_conv_channels=64,
        ff_expansion_factor=4,
        self_attention_model='rel_pos',
        n_heads=4,
        xscaling=True,
        conv_kernel_size=31,
        dropout=0.1,
        dropout_emb=0.1,
        dropout_att=0.0,
    ):
        super().__init__()

        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self.scale = math.sqrt(self.d_model)

        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None

        if subsampling:
            self.pre_encode = ConvSubsampling(
                subsampling=subsampling,
                subsampling_factor=subsampling_factor,
                feat_in=feat_in,
                feat_out=d_model,
                conv_channels=subsampling_conv_channels,
                activation=nn.ReLU(),
            )
            self.feat_out = d_model
        else:
            self.feat_out = d_model
            self.pre_encode = nn.Linear(feat_in, d_model)

        if self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model, dropout_rate=dropout, dropout_emb_rate=dropout_emb, xscale=self.xscale
            )
        elif self_attention_model == "abs_pos":
            self.pos_enc = PositionalEncoding(
                d_model=d_model, dropout_rate=dropout, max_len=6000, reverse=False, xscale=self.xscale
            )
        else:
            raise ValueError(f"Not valid self_attention_model: '{self_attention_model}'!")

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = ConformerEncoderBlock(
                d_model=d_model,
                d_ff=d_ff,
                conv_kernel_size=conv_kernel_size,
                self_attention_model=self_attention_model,
                n_heads=n_heads,
                dropout=dropout,
                dropout_att=dropout_att,
            )
            self.layers.append(layer)

        if feat_out > 0 and feat_out != self.output_dim:
            self.out_proj = nn.Linear(self.feat_out, feat_out)
            self.feat_out = feat_out
        else:
            self.out_proj = None
            self.feat_out = d_model

    @typecheck()
    def forward(self, audio_signal, length):
        audio_signal = torch.transpose(audio_signal, 1, 2)

        if isinstance(self.pre_encode, ConvSubsampling):
            audio_signal, length = self.pre_encode(audio_signal, length)
        else:
            audio_signal = self.embed(audio_signal)

        audio_signal, pos_emb = self.pos_enc(audio_signal)
        bs, xmax, idim = audio_signal.size()

        # Create the self-attention and padding masks
        pad_mask = self.make_pad_mask(length, max_time=xmax, device=audio_signal.device)
        xx_mask = pad_mask.unsqueeze(1).repeat([1, xmax, 1])
        xx_mask = xx_mask & xx_mask.transpose(1, 2)
        pad_mask = (~pad_mask).unsqueeze(2)

        for lth, layer in enumerate(self.layers):
            audio_signal = layer(x=audio_signal, att_mask=xx_mask, pos_emb=pos_emb, pad_mask=pad_mask,)

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        return audio_signal, length

    @staticmethod
    def make_pad_mask(seq_lens, max_time, device=None):
        """Make masking for padding."""
        bs = seq_lens.size(0)
        seq_range = torch.arange(0, max_time, dtype=torch.int32)
        seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_time)
        seq_lens = seq_lens.type(seq_range_expand.dtype).to(seq_range_expand.device)
        seq_length_expand = seq_lens.unsqueeze(-1)
        mask = seq_range_expand < seq_length_expand

        if device:
            mask = mask.to(device)
        return mask


class ConformerEncoderBlock(torch.nn.Module):
    """A single block of the Conformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        conv_kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
    """

    def __init__(self, d_model, d_ff, conv_kernel_size, self_attention_model, n_heads, dropout, dropout_att):
        super(ConformerEncoderBlock, self).__init__()

        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.fc_factor = 0.5

        # first feed forward module
        self.norm_feed_forward1 = LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # convolution module
        self.norm_conv = LayerNorm(d_model)
        self.conv = ConformerConvolution(d_model=d_model, kernel_size=conv_kernel_size)

        # multi-headed self-attention module
        self.norm_self_att = LayerNorm(d_model)
        if self_attention_model == 'rel_pos':
            self.self_attn = RelPositionMultiHeadAttention(n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att)
        elif self_attention_model == 'abs_pos':
            self.self_attn = MultiHeadAttention(n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att)
        else:
            raise ValueError(f"Not valid self_attention_model: '{self_attention_model}'!")

        # second feed forward module
        self.norm_feed_forward2 = LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

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
        if self.self_attention_model == 'rel_pos':
            x = self.self_attn(query=x, key=x, value=x, pos_emb=pos_emb, mask=att_mask)
        elif self.self_attention_model == 'abs_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask)
        else:
            x = None
        x = self.dropout(x) + residual

        residual = x
        x = self.norm_conv(x)
        x = self.conv(x)
        x = self.dropout(x) + residual

        residual = x
        x = self.norm_feed_forward2(x)
        x = self.feed_forward2(x)
        x = self.fc_factor * self.dropout(x) + residual

        x = self.norm_out(x)
        return x
