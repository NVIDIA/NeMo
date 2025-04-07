# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional, Union

import torch
from torch import nn

from cosmos1.models.autoregressive.modules.embedding import RotaryPositionEmbedding
from cosmos1.models.autoregressive.modules.normalization import create_norm


class Attention(nn.Module):
    """
    Attenion layer with KV cache.
    """

    def __init__(
        self,
        n_heads: int,
        n_kv_heads: Union[int, None],
        dim: int,
        max_batch_size: int,
        max_seq_len: int,
        context_dim: Optional[int] = None,
        use_qk_normalization: bool = False,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-5,
        causal_mask: Optional[bool] = True,
        head_dim: Optional[int] = None,
        fuse_qkv: bool = False,
        precision: str = "bfloat16",
        attn_type: str = "self",
    ):
        """
        Initializes the GQA module.

        Args:
            n_heads (int): The number of attention heads.
            n_kv_heads (int, optional): The number of key-value attention heads. None defaults to n_heads.
            dim (int): The dimensionality of the input and output.
            max_batch_size (int): The maximum batch size.
            max_seq_len (int): The maximum sequence length.
            context_dim (int, optional): The dimensionality of the context for cross-attn. Defaults to None.
            use_qk_normalization (bool, optional): Whether to apply QK normalization. Defaults to False.
            norm_type (str, optional): The type of normalization layer. Defaults to "rmsnorm".
            norm_eps (float, optional): The epsilon value for normalization. Defaults to 1e-5.
            causal_mask (bool, optional): Whether to use causal mask. Defaults to True.
            head_dim (int, optional): The dimensionality of each attention head. If None, defaults to dim // n_heads.
            fuse_qkv (bool, optional): Whether to fuse QKV. Defaults to False.
            precision (str, optional): The precision of the module. Defaults to "bfloat16".
            attn_type (str, optional): The type of attention. Defaults to "self".
        """
        super().__init__()
        assert attn_type in ["self", "cross", "full"], f"Invalid attention type: {attn_type}"
        self.attn_type = attn_type
        context_dim = dim if context_dim is None else context_dim

        self.dim = dim
        self.context_dim = context_dim
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_local_heads = n_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads if head_dim is None else head_dim
        self.causal_mask = causal_mask
        self.fuse_qkv = fuse_qkv
        self.precision = precision

        if fuse_qkv:
            assert context_dim == dim, f"Fuse QKV requires context_dim ({context_dim}) to be equal to dim ({dim})"
            self.total_local_head_dim = (self.n_local_heads + 2 * self.n_local_kv_heads) * self.head_dim
            self.wqkv = nn.Linear(dim, self.total_local_head_dim, bias=False)
            # Register hook to load fused QKV weights
            self._register_load_state_dict_pre_hook(self.load_hook)
        else:
            self.wq = nn.Linear(dim, self.n_local_heads * self.head_dim, bias=False)
            self.wk = nn.Linear(context_dim, self.n_local_kv_heads * self.head_dim, bias=False)
            self.wv = nn.Linear(context_dim, self.n_local_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_local_heads * self.head_dim, dim, bias=False)

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        if self.attn_type == "self":
            # Cache for key and value tensors
            self.init_kv_cache()

        # QK normalization layers
        if use_qk_normalization:
            self.q_norm = create_norm(norm_type, dim=self.head_dim, eps=norm_eps)
            self.k_norm = create_norm(norm_type, dim=self.head_dim, eps=norm_eps)

        self.use_qk_normalization = use_qk_normalization

        self.to(dtype=getattr(torch, self.precision))

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def init_kv_cache(self, dtype=None):
        cache_shape = (self.max_batch_size, self.n_local_kv_heads, self.max_seq_len, self.head_dim)
        if dtype is None:
            dtype = getattr(torch, self.precision)
        if self.attn_type == "self":
            self.cache_k = torch.zeros(cache_shape, dtype=dtype).cuda()
            self.cache_v = torch.zeros(cache_shape, dtype=dtype).cuda()

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionEmbedding,
        input_pos: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of GQA.

        Args:
            x: The input tensor of shape (batch_size, seq_len, dim).
            rope: The rotary positional embedding module.
            input_pos: The starting position of the current sequence.
            mask: The attention mask tensor.
            context: The context tensor of shape (batch_size, context_len, dim).

        Returns:
            The output tensor after applying GQA.
        """
        bsz, seqlen, _ = x.shape

        # Use one single module to handle both self-attn and cross-attn
        context = x if context is None else context
        context_len = seqlen if context is None else context.shape[1]

        if self.fuse_qkv:
            q_size = self.n_local_heads * self.head_dim
            kv_size = self.n_local_kv_heads * self.head_dim
            xq, xk, xv = self.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)
        else:
            # Compute query, key, and value projections
            xq, xk, xv = self.wq(x), self.wk(context), self.wv(context)

        # Reshape projections
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, context_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, context_len, self.n_local_kv_heads, self.head_dim)

        # QK normalization
        if self.use_qk_normalization:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        # Apply rotary positional embeddings to queries and keys
        # Only apply RoPE to self-attention!
        if self.attn_type in ["self", "full"]:
            xq, xk = rope(xq, xk, input_pos, seqlen)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))
        # xq: (bs, n_local_heads, seqlen, head_dim)
        # xk: (bs, n_kv_heads, cache_len + context_len, head_dim)
        # xv: (bs, n_kv_heads, cache_len + context_len, head_dim)
        if self.attn_type == "self":
            # Update cache with current key and value tensors
            assert input_pos is not None
            self.cache_k[:bsz, :, input_pos] = xk
            self.cache_v[:bsz, :, input_pos] = xv
            keys, values = (
                self.cache_k[:bsz, :, :],
                self.cache_v[:bsz, :, :],
            )
        else:
            keys, values = xk, xv

        # Repeat keys and values if necessary
        keys = keys.repeat_interleave(self.n_rep, dim=1)  # (bs, n_local_heads, cache_len + context_len, head_dim)
        values = values.repeat_interleave(self.n_rep, dim=1)  # (bs, n_local_heads, cache_len + context_len, head_dim)

        # For self-attention, `is_causal` should be set to False when KV cache is pre-computed and used,
        # since the masking is handled outside this attention module.
        # For cross-attention, it's always full-attn without causal mask
        is_causal = False
        output = scaled_dot_product_attention(
            xq,
            keys,
            values,
            head_dim=self.head_dim,
            mask=mask,
            is_causal=is_causal,
            dropout_p=0.0,
        )
        output = output.view(bsz, seqlen, -1)
        output = self.wo(output)
        return output


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_dim: int,
    mask: Optional[torch.Tensor] = None,
    is_causal: Optional[bool] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    PyTorch's native implementation of Flash Attention 2.

    If `is_causal` is given, then the causal attention mask is applied accordingly:
    - If `is_causal` is True, the standard upper-left causal attention masking is applied.
    - If `is_causal` is False, no attention mask is applied, unless an explicit mask tensor is
      provided (i.e., `mask is not None`).

    If `is_causal` is not given (i.e., `is_causal is None`), then the attention mask is applied
    based on the provided mask tensor:
    - If no explicit attention mask is given (i.e., `mask is None`), `is_causal` is set to True,
    leading to the standard upper-left causal attention masking.
    - If an attention mask is given (i.e., `mask is not None`), the provided mask is used,
    and `is_causal` is set to False.

    Args:
        q (torch.Tensor): Query tensor
        k (torch.Tensor): Key tensor
        v (torch.Tensor): Value tensor
        head_dim (int): Dimension of each attention head
        mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
        is_causal (Optional[bool], optional): Whether to apply causal attention mask. Defaults to None.
        dropout_p (float, optional): Dropout rate. Defaults to 0.0.

    Returns:
        torch.Tensor: Output tensor after applying scaled dot-product attention
    """
    scale = 1.0 / math.sqrt(head_dim)
    if is_causal is None:
        is_causal = mask is None
    y = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=dropout_p,
        scale=scale,
        is_causal=is_causal,
    )
    return y.transpose(1, 2).contiguous()
