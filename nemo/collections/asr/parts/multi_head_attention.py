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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
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
Part of this code is adopted from https://github.com/espnet/espnet
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'RelPositionMultiHeadAttention',
    'RelPositionalEncoding',
    'PositionalEncoding',
    'RelPositionMultiHeadAttention2' 'RelPositionalEncoding2',
]


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer.
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
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
        """Transforms query, key and value.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value (torch.Tensor): (batch, time2, size)
        returns:
            q (torch.Tensor): (batch, head, time1, size)
            k (torch.Tensor): (batch, head, time2, size)
            v (torch.Tensor): (batch, head, time2, size)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        Args:
            value (torch.Tensor): (batch, time2, size)
            scores(torch.Tensor): (batch, time1, time2)
            mask(torch.Tensor): (batch, time1, time2)
        returns:
            value (torch.Tensor): transformed `value` (batch, time2, d_model) weighted by the attention scores
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # .eq(0)  # (batch, 1, time1, time2)
            if scores.dtype == torch.float16:
                dtype = np.float16
            else:
                dtype = np.float32
            min_value = np.finfo(dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask, pos_emb=None):
        """Compute 'Scaled Dot Product Attention'.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
        returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadAttention(MultiHeadAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate, pos_bias_u, pos_bias_v):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            # nn.init.normal_(self.pos_bias_u, 0.0, 0.02)
            # nn.init.normal_(self.pos_bias_v, 0.0, 0.02)
            nn.init.zeros_(self.pos_bias_u)
            nn.init.zeros_(self.pos_bias_v)

        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v

    # buggy one
    # def rel_shift(self, x, zero_triu=False):
    #     """Compute relative positinal encoding.
    #     Args:
    #         x (torch.Tensor): (batch, time, size)
    #         zero_triu (bool): return the lower triangular part of the matrix
    #     """
    #     zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
    #     x_padded = torch.cat([zero_pad, x], dim=-1)
    #
    #     x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
    #     x = x_padded[:, :, 1:].view_as(x)
    #
    #     x = x.squeeze(0)
    #     x = torch.tril(x) + torch.triu(x.transpose(1,2), diagonal=1)
    #     x = x.unsqueeze(0)
    #     if zero_triu:
    #         ones = torch.ones((x.size(2), x.size(3)))
    #         x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]
    #
    #     return x

    def rel_shift(self, x):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): (batch, nheads, time, 2*time-1)
        """
        qlen = x.size(2)
        pos_len = x.size(-1)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.nn.functional.pad(x, pad=(0, qlen))
        x = x.view(x.size(0), x.size(1), qlen, pos_len + 1)
        return x[:, :, :, 0:qlen].flip(dims=[-1])

    def forward(self, query, key, value, mask, pos_emb):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            pos_emb (torch.Tensor) : (batch, time1, size)
        Returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
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
    Args:
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        reverse (int): whether to reverse the input position
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False, xscale=None):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = xscale
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.reverse:
            needed_size = 2 * x.size(1) - 1
        else:
            needed_size = x.size(1)
        if self.pe is not None:
            if self.pe.size(1) >= needed_size:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(needed_size, self.d_model)
        if self.reverse:
            position = torch.arange(-(x.size(1) - 1), x.size(1), 1.0, dtype=torch.float32).unsqueeze(1)
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
            Encoded Output (torch.Tensor): Its shape is (batch, time, ...)
        """
        self.extend_pe(x)
        if self.xscale:
            x = x * self.xscale
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x), None


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, xscale=None, dropout_emb_rate=0.0):
        super().__init__(d_model, dropout_rate, max_len, reverse=True, xscale=xscale)

        if dropout_emb_rate > 0:
            self.dropout_emb = nn.Dropout(dropout_emb_rate)
        else:
            self.dropout_emb = None

        self.max_len = max_len

    def forward(self, x):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
        Returns:
            x (torch.Tensor): Its shape is (batch, time, ...)
            pos_emb (torch.Tensor): Its shape is (1, time, ...)
        """
        self.extend_pe(x)
        if self.xscale:
            x = x * self.xscale

        start_pos = (self.pe.size(1) + 1) // 2 - x.size(1)
        pos_emb = self.pe[:, start_pos:-start_pos]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb


# New ones
class RelPositionMultiHeadAttention2(nn.Module):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate, pos_bias_u, pos_bias_v):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__()

        # linear transformation for positional encoding
        self.d_head = n_feat // n_head
        self.n_head = n_head
        self.d_model = n_feat
        self.qkv_net = nn.Linear(n_feat, 3 * n_head * self.d_head, bias=False)
        self.o_net = nn.Linear(n_head * self.d_head, n_feat, bias=False)

        if pos_bias_u is None or pos_bias_v is None:
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            # nn.init.normal_(self.r_r_bias, 0.0, 0.02)
            # nn.init.normal_(self.r_w_bias, 0.0, 0.02)
            nn.init.zeros_(self.r_r_bias)
            nn.init.zeros_(self.r_w_bias)
        else:
            self.r_r_bias = pos_bias_u
            self.r_w_bias = pos_bias_v

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
        self.scale = math.sqrt(self.d_head)
        self.dropout = nn.Dropout(p=dropout_rate)

    def rel_shift(self, x):
        # x: (qlen x klen x bsz x n_head)
        # batch, nheads, time, 2 * time - 1
        # x = x.permute(2, 3, 0, 1)
        qlen = x.size(2)
        pos_len = x.size(-1)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.nn.functional.pad(x, pad=(0, qlen))
        x = x.view(x.size(0), x.size(1), qlen, pos_len + 1)
        x = x[:, :, :, 0:qlen].flip(dims=[-1])
        # x = x.permute(2, 3, 0, 1)
        return x
        # buggy code
        # zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        # zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        # x_padded = torch.cat([zero_pad, x], dim=1)
        #
        # x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        # x_padded = x_padded.view(*x_padded_shape)
        #
        # x = x_padded[1:].view_as(x)
        #
        # x = x.permute(2, 3, 0, 1).squeeze(0)
        # x = torch.tril(x) + torch.triu(x.transpose(1, 2), diagonal=1)
        # x = x.permute(1, 2, 0).unsqueeze(2)
        # return x

    def forward(self, query, key, value, mask, pos_emb):
        # key and values are ignored
        # query :(qlen, batch)
        w = query.transpose(0, 1)
        r = pos_emb.squeeze(0).unsqueeze(1)

        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        w_heads = self.qkv_net(w)
        r_head_k = self.r_net(r)

        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + self.r_w_bias  # qlen x bsz x n_head x d_head
        # AC = torch.einsum("ibnd,jbnd->ijbn", (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head
        AC = torch.einsum("ibnd,jbnd->bnij", (rw_head_q, w_head_k))  # bsz x n_head x qlen x klen

        rr_head_q = w_head_q + self.r_r_bias
        # BD = torch.einsum("ibnd,jnd->ijbn", (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        BD = torch.einsum('ibnd,jnd->bnij', (rr_head_q, r_head_k))  # bsz x n_head x qlen x klen

        BD = self.rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        # new dim: [bsz x n_head x qlen x klen]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # attn_mask = mask.transpose(0, 2)

        # attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None], -1e30).type_as(attn_score)
        if attn_score.dtype == torch.float16:
            dtype = np.float16
        else:
            dtype = np.float32
        min_value = np.finfo(dtype).min

        attn_score = attn_score.masked_fill(mask[:, None, :, :], min_value)

        # attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = F.softmax(attn_score, dim=-1).masked_fill(mask[:, None, :, :], 0.0)

        attn_prob = self.dropout(attn_prob)

        # attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_head_v))
        attn_vec = torch.einsum("bnij,jbnd->bind", (attn_prob, w_head_v))

        # attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        attn_out = self.o_net(attn_vec)

        # attn_out = attn_out.transpose(0, 1).contiguous()
        return attn_out


class RelPositionalEncoding2(nn.Module):
    def __init__(self, d_model, dropout_rate, max_len=None, xscale=None, dropout_emb_rate=0.0):
        super().__init__()

        self.demb = d_model
        demb = d_model

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

        if dropout_emb_rate > 0:
            self.dropout_emb = nn.Dropout(dropout_emb_rate)
        else:
            self.dropout_emb = None

        self.xscale = xscale
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor):
        klen = x.size(1)
        # pos_seq = torch.arange(klen - 1, -1, -1.0, device=x.device, dtype=x.dtype)
        pos_seq = torch.arange(-(klen - 1), (klen - 1) + 1, 1.0, device=x.device, dtype=x.dtype)
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        if self.xscale:
            x = x * self.xscale

        return self.dropout(x), pos_emb[None, :, :]
