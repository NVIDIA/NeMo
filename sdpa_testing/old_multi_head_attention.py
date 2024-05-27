import math
from functools import lru_cache
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.utils import avoid_float16_autocast_context


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer of Transformer.
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate, max_cache_len=0):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadAttention, self).__init__()
        self.cache_drop_size = None
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

        self._max_cache_len = max_cache_len

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
            mask = mask.unsqueeze(1)  # (batch, 1, time1, time2)
            scores = scores.masked_fill(mask, -10000.0)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask, pos_emb=None, cache=None):
        """Compute 'Scaled Dot Product Attention'.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            cache (torch.Tensor) : (batch, time_cache, size)

        returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
            cache (torch.Tensor) : (batch, time_cache_next, size)
        """
        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)

        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        # temporary until we solve this more gracefully
        with avoid_float16_autocast_context():
            q, k, v = self.forward_qkv(query, key, value)
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.s_d_k
            out = self.forward_attention(v, scores, mask)
        if cache is None:
            return out
        else:
            return out, cache

    def update_cache(self, key, value, query, cache):
        if cache is not None:
            key = value = torch.cat([cache, key], dim=1)
            q_keep_size = query.shape[1] - self.cache_drop_size
            cache = torch.cat([cache[:, q_keep_size:, :], query[:, :q_keep_size, :]], dim=1)
        return key, value, query, cache


class RelPositionMultiHeadAttention(MultiHeadAttention):
    """Multi-Head Attention layer of Transformer-XL with support of relative positional encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate, pos_bias_u, pos_bias_v, max_cache_len=0):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head=n_head, n_feat=n_feat, dropout_rate=dropout_rate, max_cache_len=max_cache_len)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable biases are used in matrix c and matrix d
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

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x (torch.Tensor): (batch, nheads, time, 2*time-1)
        """
        b, h, qlen, pos_len = x.size()  # (b, h, t1, t2)
        # need to add a column of zeros on the left side of last dimension to perform the relative shifting
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t1, t2)
        return x

    def forward(self, query, key, value, mask, pos_emb, cache=None):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            pos_emb (torch.Tensor) : (batch, time1, size)
            cache (torch.Tensor) : (batch, time_cache, size)

        Returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
            cache (torch.Tensor) : (batch, time_cache_next, size)
        """
        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)

        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        # temporary until we solve this more gracefully
        with avoid_float16_autocast_context():
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
            # drops extra elements in the matrix_bd to match the matrix_ac's size
            matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]

            scores = (matrix_ac + matrix_bd) / self.s_d_k  # (batch, head, time1, time2)

            out = self.forward_attention(v, scores, mask)

        if cache is None:
            return out
        else:
            return out, cache
