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

from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
)
from nemo.collections.asr.parts.utils.activations import Swish
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerFeedForward

__all__ = ['TransformerEncoderLayer']


class TransformerEncoderLayer(torch.nn.Module):
    """A single block of the transformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        ff_activation : activation function used in FF network
    """

    def __init__(
        self,
        d_model,
        d_ff,
        self_attention_model='abs_pos',
        n_heads=1,
        dropout=0.0,
        dropout_ff=0.0,
        dropout_att=0.0,
        pos_bias_u=None,
        pos_bias_v=None,
        pre_norm = False,
        activation_ff = nn.ReLU(),
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.pre_norm = pre_norm
        self.norm_self_att = LayerNorm(d_model, eps=1e-5)
       
        # multi-headed self-attention module
        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        if self_attention_model == 'rel_pos':
            self.self_attn = RelPositionMultiHeadAttention(
                n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att, pos_bias_u=pos_bias_u, pos_bias_v=pos_bias_v
            )
        elif self_attention_model == 'abs_pos':
            self.self_attn = MultiHeadAttention(n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att)
        else:
            raise ValueError(
                f"'{self_attention_model}' is not not a valid value for 'self_attention_model', "
                f"valid values can be from ['rel_pos', 'abs_pos']"
            )

        self.norm_feed_forward = LayerNorm(d_model, eps=1e-5)

        # feed forward module
        self.feed_forward = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout_ff, activation=activation_ff)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None):
        """
        Args:
            x (torch.Tensor): input signals (B, L, H)
            att_mask (torch.Tensor): attention masks(B, L, L)
            pos_emb (torch.Tensor): (L, 1, H)
            pad_mask (torch.tensor): padding mask
        Returns:
            x (torch.Tensor): (B, L, H)
        """
        
        # MHA block
        if self.pre_norm:
            x1 = self.norm_self_att(x)
        else:
            x1 = x
        
        if self.self_attention_model == 'rel_pos':
            x1 = self.self_attn(query=x1, key=x1, value=x1, mask=att_mask, pos_emb=pos_emb)
        elif self.self_attention_model == 'abs_pos':
            x1 = self.self_attn(query=x1, key=x1, value=x1, mask=att_mask)
        else:
            x1 = None
        
        # add and norm
        x = x + self.dropout1(x1)
        if not self.pre_norm:
            x = self.norm_self_att(x)

        # FF block
        if self.pre_norm:
            x1 = self.norm_feed_forward(x)
        else:
            x1 = x
        
        x1 = self.feed_forward(x1)

        # add and norm
        x = x + self.dropout2(x1)
        if not self.pre_norm:
            x = self.norm_feed_forward(x)

        return x