# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Copyright 2020 Hirofumi Inaguma(Kyoto University)
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

"""
Part of this code is adopted from https://github.com/espnet/espnet and https://github.com/hirofumi0810/neural_sp/
"""

import math

import numpy as np
import torch
import torch.nn as nn

from nemo.utils import logging

__all__ = [
    'RelPositionMultiHeadAttention_old',
    'RelPositionMultiHeadAttention',
    'RelPositionalEncoding_old',
    'RelPositionalEncoding',
    'PositionalEncoding',
]


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer.
    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.
        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :return torch.Tensor transformed query, key and value
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor scores: (batch, time1, time2)
        :param torch.Tensor mask: (batch, time1, time2)
        :return torch.Tensor transformed `value` (batch, time2, d_model)
            weighted by the attention score (batch, time1, time2)
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'.
        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadAttention(MultiHeadAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional ecoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu=False):
        """Compute relative positinal encoding.
        :param torch.Tensor x: (batch, time, size)
        :param bool zero_triu: return the lower triangular part of the matrix
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, mask, pos_emb):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor pos_emb: (batch, time1, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.
    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    :param reverse: whether to reverse the input position
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False, xscale=None):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = xscale  # math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(x.size(1) - 1, -1, -1.0, dtype=torch.float32).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
        """
        self.extend_pe(x)
        if self.xscale:
            x = x * self.xscale
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x), None


class RelPositionalEncoding(PositionalEncoding):
    """Relitive positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, dropout_emb_rate=0.0, xscale=None):
        """Initialize class.
        :param int d_model: embedding dim
        :param float dropout_rate: dropout rate
        :param int max_len: maximum input length
        """
        super().__init__(d_model, dropout_rate, max_len, reverse=True, xscale=xscale)

        if dropout_emb_rate > 0:
            self.dropout_emb = nn.Dropout(dropout_emb_rate)
        else:
            self.dropout_emb = None

    def forward(self, x):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
        Returns:
            torch.Tensor: x. Its shape is (batch, time, ...)
            torch.Tensor: pos_emb. Its shape is (1, time, ...)
        """
        self.extend_pe(x)
        if self.xscale:
            x = x * self.xscale
        pos_emb = self.pe[:, : x.size(1)]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb


class RelPositionMultiHeadAttention_old(nn.Module):
    """Relative multi-head attention layer for TransformerXL.
    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        adim: (int) dimension of attention space
        odim: (int) dimension of output
        n_heads (int): number of heads
        dropout (float): dropout probability for attenion weights
        dropout_head (float): HeadDrop probability
        bias (bool): use bias term in linear layers
        param_init (str): parameter initialization method
        xl_like (bool): use TransformerXL like relative positional encoding.
            Otherwise, use relative positional encoding like Shaw et al. 2018
    """

    def __init__(
        self, kdim, qdim, adim, odim, n_heads, dropout, dropout_head=0.0, bias=False, param_init='', xl_like=False
    ):

        super().__init__()

        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(self.d_k)
        self.xl_like = xl_like

        self.dropout_attn = nn.Dropout(p=dropout)
        self.dropout_head = dropout_head

        assert kdim == qdim
        # NOTE: relative attention is supprted for self-attention only
        self.w_key = nn.Linear(kdim, adim, bias=bias)
        self.w_value = nn.Linear(kdim, adim, bias=bias)
        self.w_query = nn.Linear(qdim, adim, bias=bias)
        self.w_out = nn.Linear(adim, odim, bias=bias)

        if xl_like:
            self.w_pos = nn.Linear(qdim, adim, bias=bias)

        if param_init == 'xavier_uniform':
            self.reset_parameters(bias)
        else:
            logging.info('Parameter initialization for RelativeMultiheadAttentionMechanism skipped.')

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        # NOTE: see https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_value.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_key.bias, 0.0)
            nn.init.constant_(self.w_value.bias, 0.0)
            nn.init.constant_(self.w_query.bias, 0.0)

        nn.init.xavier_uniform_(self.w_out.weight)
        if bias:
            nn.init.constant_(self.w_out.bias, 0.0)

        if self.xl_like:
            nn.init.xavier_uniform_(self.w_pos.weight)
            if bias:
                nn.init.constant_(self.w_pos.bias, 0.0)

    def _rel_shift(self, xs):
        """Calculate relative positional attention efficiently.
        Args:
            xs (FloatTensor): `[B, qlen, klen, H]`
        Returns:
            xs_shifted (FloatTensor): `[B, qlen, klen, H]`
        """
        bs, qlen, klen, n_heads = xs.size()
        # `[qlen, klen, B, H]` -> `[B, qlen, klen, H]`
        xs = xs.permute(1, 2, 0, 3).contiguous().view(qlen, klen, bs * n_heads)

        zero_pad = xs.new_zeros((qlen, 1, bs * n_heads))
        xs_shifted = torch.cat([zero_pad, xs], dim=1).view(klen + 1, qlen, bs * n_heads)[1:].view_as(xs)
        return xs_shifted.view(qlen, klen, bs, n_heads).permute(2, 0, 1, 3)

    def forward(self, query, key, pos_emb, mask, u_bias=None, v_bias=None):
        """Forward pass.
        Args:
            cat (FloatTensor): `[B, mlen+qlen, kdim]`
            mask (ByteTensor): `[B, qlen, mlen+qlen]`
            pos_emb (LongTensor): `[qlen, 1, d_model]`
            u_bias (nn.Parameter): `[H, d_k]`
            v_bias (nn.Parameter): `[H, d_k]`
        Returns:
            cv (FloatTensor): `[B, qlen, vdim]`
            aw (FloatTensor): `[B, H, qlen, mlen+qlen]`
        """
        bs, qlen = query.size()[:2]
        mlen = key.size(1) - qlen
        # NOTE: cat already includes memory, i.e., klen=mlen+qlen

        if mask is not None:
            mask = mask.unsqueeze(3).repeat([1, 1, 1, self.n_heads])
            assert mask.size() == (bs, qlen, mlen + qlen, self.n_heads), (
                mask.size(),
                (bs, qlen, mlen + qlen, self.n_heads),
            )

        k = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)  # `[B, mlen+qlen, H, d_k]`
        v = self.w_value(key).view(bs, -1, self.n_heads, self.d_k)  # `[B, mlen+qlen, H, d_k]`
        q = self.w_query(key[:, -qlen:]).view(bs, -1, self.n_heads, self.d_k)  # `[B, qlen, H, d_k]`

        if self.xl_like:
            _pos_embs = self.w_pos(pos_emb)
        else:
            _pos_embs = self.w_value(pos_emb)
        _pos_embs = _pos_embs.view(-1, self.n_heads, self.d_k)  # `[mlen+qlen, H, d_k]`

        # content-based attention term: (a) + (c)
        if u_bias is not None:
            assert self.xl_like
            AC = torch.einsum("bihd,bjhd->bijh", ((q + u_bias[None, None]), k))  # `[B, qlen, mlen+qlen, H]`
        else:
            AC = torch.einsum("bihd,bjhd->bijh", (q, k))  # `[B, qlen, mlen+qlen, H]`

        # position-based attention term: (b) + (d)
        if v_bias is not None:
            assert self.xl_like
            BD = torch.einsum("bihd,jhd->bijh", ((q + v_bias[None, None]), _pos_embs))  # `[B, qlen, mlen+qlen, H]`
        else:
            BD = torch.einsum("bihd,jhd->bijh", (q, _pos_embs))  # `[B, qlen, mlen+qlen, H]`

        # Compute positional attention efficiently
        BD = self._rel_shift(BD)

        # the attention is the sum of content-based and position-based attention
        e = (AC + BD) / self.scale  # `[B, qlen, mlen+qlen, H]`

        # Compute attention weights
        if mask is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=e.dtype).numpy().dtype).min)
            e = e.masked_fill_(mask == 0, NEG_INF)  # `[B, qlen, mlen+qlen, H]`
        aw = torch.softmax(e, dim=2)
        aw = self.dropout_attn(aw)  # `[B, qlen, mlen+qlen, H]`

        # mask out each head independently (HeadDrop)
        if self.dropout_head > 0 and self.training:
            aw_masked = aw.clone()
            aw_masked = aw_masked.permute(0, 3, 1, 2)
            aw_masked = self.dropout_head(aw_masked, self.n_heads, self.dropout_head)  # `[B, H, qlen, klen]`
            aw_masked = aw_masked.permute(0, 2, 3, 1)

        cv = torch.einsum("bijh,bjhd->bihd", (aw, v))  # `[B, qlen, H, d_k]`
        cv = cv.contiguous().view(bs, -1, self.n_heads * self.d_k)  # `[B, qlen, H * d_k]`
        cv = self.w_out(cv)
        # aw = aw.permute(0, 3, 1, 2)  # `[B, H, qlen, mlen+qlen]`

        return cv  # , aw


class RelPositionalEncoding_old(nn.Module):
    def __init__(self, d_model, dropout, dropout_emb_rate=0.0, xscale=None):
        """Positional embedding for TransformerXL."""
        super().__init__()
        self.d_model = d_model
        self.xscale = xscale
        inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer("inv_freq", inv_freq)

        self.dropout = nn.Dropout(p=dropout)

        if dropout_emb_rate > 0:
            self.dropout_emb = nn.Dropout(dropout_emb_rate)
        else:
            self.dropout_emb = None

        self.xscale = 1  # math.sqrt(self.d_model)

    def forward(self, xs, mlen=0, clamp_len=-1, zero_center_offset=False):
        """Forward pass.
        Args:
            xs (FloatTensor): `[B, L, d_model]`
            mlen (int); length of memory
            clamp_len (int):
            zero_center_offset (bool):
        Returns:
            pos_emb (LongTensor): `[L, 1, d_model]`
        """
        if zero_center_offset:
            pos_idxs = torch.arange(mlen - 1, -xs.size(1) - 1, -1.0, dtype=torch.float, device=xs.device)
        else:
            pos_idxs = torch.arange(mlen + xs.size(1) - 1, -1, -1.0, dtype=torch.float, device=xs.device)

        # truncate by maximum length
        if clamp_len > 0:
            pos_idxs.clamp_(max=clamp_len)

        # outer product
        sinusoid_inp = torch.einsum("i,j->ij", pos_idxs, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        if self.xscale:
            xs = self.xscale * xs

        return self.dropout(xs), pos_emb.unsqueeze(1)
