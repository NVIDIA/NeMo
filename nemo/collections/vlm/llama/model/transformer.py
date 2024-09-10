# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import fairscale.nn.model_parallel.initialize as fs_init

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)

from PIL import Image as PIL_Image

from torch import nn
from torch.distributed import _functional_collectives as funcol

from nemo.collections.vlm.llama.model.encoder_utils import (
    build_encoder_attention_mask,
    contract_num_tokens_from_mult8,
    expand_num_tokens_to_mult8,
    initialize_global_position_embedding_from_local,
    resize_global_position_embedding,
    resize_local_position_embedding,
)
from nemo.collections.vlm.llama.utils import get_negative_inf_value, to_2tuple


logger = logging.getLogger(__name__)
MP_SCALE = 8


def reduce_from_tensor_model_parallel_region(input_):
    """All-reduce the input tensor across model parallel group."""
    output = funcol.all_reduce(input_, "sum", group=fs_init.get_model_parallel_group())
    output = funcol.wait_tensor(output)
    return output


def gather_from_tensor_model_parallel_region(input_):
    """Gather tensors and concatenate along the last dimension."""

    world_size = fs_init.get_model_parallel_world_size()
    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = fs_init.get_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    output = funcol.all_gather_tensor(
        input_,
        gather_dim=last_dim,
        group=fs_init.get_model_parallel_group(),
    )
    output = funcol.wait_tensor(output)
    return output




def _get_full_row_masked_out_mask(
    attn_bias,
    negative_inf_value,
):
    """
    attn_bias should be a 4D tensor of shape [B, H, S1, S2]
    where B is the batch size, H is the number of heads,
    and S1/S2 are the sequence lengths. This returns
    a 4D tensor of shape [B, H, S1, 1] which stores boolean
    values which are 0 if the a full row in the last dimension
    contains negative infinity values, otherwise it's 1.
    """
    return (attn_bias != negative_inf_value).any(dim=-1).type_as(attn_bias)[..., None]


def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)



class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# Image encoder for inference
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x



class ColumnParallelConv2dPatch(torch.nn.Module):
    """Conv2D Patching layer with model parallelism.
    Column parallel over unfolded input.
    Arguments:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Size of convolution kernel.
        stride (default 1): Stride for convolution.
        bias (default False): Use bias in Conv2d.
    Input: (bsz, in_channels, width, height)
    Output: (bsz, num_tokens, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        bias: Optional[bool] = False,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self._unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride)
        self._linear = ColumnParallelLinear(
            in_channels * kernel_size[0] * kernel_size[1],
            out_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._unfold(x)
        x = x.permute(0, 2, 1)
        x = F.linear(x, self._linear.weight)
        x = gather_from_tensor_model_parallel_region(x)
        return x


class ImageFeedForward(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float,
        act_layer: Callable = nn.GELU,
    ):
        super().__init__()
        # layers
        self.c_fc = ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=True,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.c_proj = RowParallelLinear(
            hidden_dim,
            dim,
            bias=True,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        self.non_linearity = act_layer()
        self.dropout = dropout

    def forward(self, x):
        hidden = F.linear(x, self.c_fc.weight, self.c_fc.bias)
        hidden = self.non_linearity(hidden)
        hidden = F.linear(hidden, self.c_proj.weight)
        hidden = reduce_from_tensor_model_parallel_region(hidden)
        hidden += self.c_proj.bias
        return hidden


class ImageAttention(nn.Module):
    def __init__(
        self,
        dim,
        head_dim,
        n_heads,
    ):
        super().__init__()
        model_parallel_size = fs_init.get_model_parallel_world_size()
        qkvo_replication = 1
        if model_parallel_size > 16:
            qkvo_replication = model_parallel_size // 8

        self.n_kv_heads = n_heads
        self.n_local_heads = n_heads * qkvo_replication // model_parallel_size
        self.n_local_kv_heads = (
            self.n_kv_heads * qkvo_replication // model_parallel_size
        )
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.wq = ColumnParallelLinear(
            dim,
            qkvo_replication * n_heads * self.head_dim,
            bias=True,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            dim,
            qkvo_replication * self.n_kv_heads * self.head_dim,
            bias=True,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            dim,
            qkvo_replication * self.n_kv_heads * self.head_dim,
            bias=True,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            qkvo_replication * n_heads * self.head_dim,
            dim,
            bias=True,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        self.qkvo_replication = qkvo_replication

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ):

        xq, xk, xv = [
            F.linear(x, w, b)
            for (w, b) in [
                (self.wq.weight, self.wq.bias),
                (self.wk.weight, self.wk.bias),
                (self.wv.weight, self.wv.bias),
            ]
        ]

        bs, slen, _ = xq.shape

        xq = xq.view(bs, slen, self.n_local_heads, self.head_dim)
        xk = xk.view(bs, xk.shape[1], self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs, xv.shape[1], self.n_local_kv_heads, self.head_dim)

        xq, xk, xv = [tensor.transpose(1, 2) for tensor in (xq, xk, xv)]

        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)

        attn_output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=mask, dropout_p=0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bs, slen, -1)

        out = F.linear(attn_output, self.wo.weight)
        out = reduce_from_tensor_model_parallel_region(out)
        out = out / self.qkvo_replication
        out += self.wo.bias
        return out


class ImageTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        gated: bool = False,
    ):
        super().__init__()
        assert d_model % n_head == 0
        self.n_heads = n_head
        self.head_dim = d_model // self.n_heads
        self.attn = ImageAttention(
            dim=d_model,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
        )
        self.ln_1 = LayerNorm(d_model)
        self.mlp = ImageFeedForward(
            dim=d_model,
            hidden_dim=int(mlp_ratio * d_model),
            dropout=0.0,
            act_layer=act_layer,
        )
        self.ln_2 = LayerNorm(d_model)
        self.gated = gated
        if gated:
            self.gate_attn = nn.Parameter(torch.zeros(1))
            self.gate_ffn = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        _gate_attn = 1 if not self.gated else self.gate_attn.tanh()
        _gate_ffn = 1 if not self.gated else self.gate_ffn.tanh()
        x = x + _gate_attn * self.attn(self.ln_1(x), mask=mask)
        x = x + _gate_ffn * self.mlp(self.ln_2(x))
        return x


class ImageTransformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        gated: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ImageTransformerBlock(
                    d_model=width,
                    n_head=heads,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    gated=gated,
                )
                for _ in range(self.layers)
            ]
        )

    def forward(self, x: torch.Tensor, return_intermediate=None, mask=None):
        out = []
        for idx, r in enumerate(self.resblocks):
            if return_intermediate is not None and idx in return_intermediate:
                out.append(x)
            x = r(x, mask=mask)
        if return_intermediate is not None:
            return x, torch.stack(out, dim=-1)
        return x


class VisionEncoder(nn.Module):
    def __init__(
        self,
        max_num_tiles: int,
        ckpt_path: str = None,
        image_size: int = 224,
        patch_size: int = 14,
        width: int = 1280,
        layers: int = 32,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        in_channels: int = 3,
        load_ckpt: bool = False,
        n_global_layers: int = 2,
        global_model: bool = False,
        return_intermediate=None,
    ):
        super().__init__()
        self.global_model = global_model
        self.return_intermediate = return_intermediate
        self.max_num_tiles = max_num_tiles
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        self.conv1 = ColumnParallelConv2dPatch(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width)
        )
        self.ln_post = LayerNorm(width)
        self.ln_pre = LayerNorm(width)
        self.transformer = ImageTransformer(
            width, layers, heads, mlp_ratio, act_layer=act_layer
        )
        # pre and post tile position embedding
        self.global_transformer = ImageTransformer(
            width, n_global_layers, heads, mlp_ratio, act_layer=act_layer, gated=True
        )
        # pre and post tile position embedding
        self.pre_tile_pos_embed = TilePositionEmbedding(
            num_tiles=max_num_tiles,
            width=width,
            gated=True,
        )
        self.post_tile_pos_embed = TilePositionEmbedding(
            num_tiles=max_num_tiles,
            width=width,
            gated=True,
        )
        self.gated_positional_embedding = nn.Parameter(
            scale
            * torch.randn(
                max_num_tiles,
                max_num_tiles,
                self.grid_size[0] * self.grid_size[1] + 1,
                width,
            )
        )
        self.gated_positional_embedding_gate = nn.Parameter(torch.zeros(1))

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool = True,
        missing_keys: List[str] = None,
        unexpected_keys: List[str] = None,
        error_msgs: List[str] = None,
        return_state_dict: bool = False,
    ) -> None:
        orig_pos_embed = state_dict.get(prefix + "positional_embedding")
        if orig_pos_embed is not None:
            new_pos_embed = resize_local_position_embedding(
                orig_pos_embed, self.grid_size
            )
            state_dict[prefix + "positional_embedding"] = new_pos_embed
        if hasattr(self, "gated_positional_embedding"):
            if prefix + "gated_positional_embedding" not in state_dict:
                # resize positional_embedding to fit the new grid size
                global_pos_embed = initialize_global_position_embedding_from_local(
                    new_pos_embed,
                    self.grid_size,
                    self.max_num_tiles,
                    self.max_num_tiles,
                )
                state_dict[prefix + "gated_positional_embedding"] = global_pos_embed
                state_dict[prefix + "gated_positional_embedding_gate"] = torch.zeros(
                    1, dtype=global_pos_embed.dtype
                )
                logger.info(
                    f"Initialized global positional embedding with size {global_pos_embed.size()}"
                )
            else:
                global_pos_embed = resize_global_position_embedding(
                    state_dict[prefix + "gated_positional_embedding"],
                    self.grid_size,
                    self.max_num_tiles,
                    self.max_num_tiles,
                )
                logger.info(
                    f"Resized global positional embedding from {state_dict[prefix + 'gated_positional_embedding'].size()} to {global_pos_embed.size()}"
                )
                state_dict[prefix + "gated_positional_embedding"] = global_pos_embed
        if return_state_dict:
            return state_dict

    def apply_positional_embedding(self, x, ar):
        out = []
        # apply regular position embedding
        bsz, num_chunks, num_tokens, dim = x.shape
        x = x.view(bsz * num_chunks, num_tokens, dim)
        x = x + self.positional_embedding * (
            1 - self.gated_positional_embedding_gate.tanh()
        )
        x = x.view(bsz, num_chunks, num_tokens, dim)
        for idx, arx in enumerate(ar):
            _pos_embed = self.gated_positional_embedding[: arx[0], : arx[1]]
            _pos_embed = _pos_embed.reshape(arx[0] * arx[1], *_pos_embed.shape[2:])
            x[idx, : arx[0] * arx[1]] += (
                _pos_embed * self.gated_positional_embedding_gate.tanh()
            )
        return x

    def apply_class_embedding(self, x):
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        return x

    def forward(self, images: torch.Tensor, ar: torch.Tensor) -> torch.Tensor:
        if images.ndim == 5:
            num_concurrent_media = 1
            bsz, num_chunks, nch, w, h = images.shape
        else:
            bsz, num_concurrent_media, num_chunks, nch, w, h = images.shape

        images = images.reshape(bsz * num_concurrent_media * num_chunks, nch, w, h)
        ar = ar.reshape(bsz * num_concurrent_media, 2)

        # patch embedding
        x = images.reshape(bsz * num_concurrent_media * num_chunks, nch, w, h)
        x = self.conv1(x)  # shape = [*, width, grid ** 2]
        _, ntok, dim = x.shape
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok, dim)

        # tile embeddings
        x = self.pre_tile_pos_embed(x, ar)
        x = x.reshape(bsz * num_concurrent_media * num_chunks, ntok, dim)

        # apply cls token
        x = self.apply_class_embedding(x)
        ntok += 1

        # apply position embeddings
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok, dim)
        x = self.apply_positional_embedding(x, ar)

        x = self.ln_pre(x)
        npad, attn_mask = 0, None
        x, npad = expand_num_tokens_to_mult8(x)
        attn_mask = build_encoder_attention_mask(x, ar, ntok, num_chunks, 1)
        x = x.view(bsz * num_concurrent_media, -1, dim)
        x, int_x = self.transformer(
            x, return_intermediate=self.return_intermediate, mask=attn_mask
        )

        x = self.ln_post(x)
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, dim)
        x = self.post_tile_pos_embed(x, ar)
        x = x.reshape(bsz * num_concurrent_media, num_chunks * (ntok + npad), dim)
        x = self.global_transformer(x, mask=attn_mask)
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, dim)
        x = contract_num_tokens_from_mult8(x, npad)

        # adding back intermediate layer outputs
        x = x.reshape(bsz, num_concurrent_media, num_chunks, ntok, dim)
        int_x = int_x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, -1)
        int_x = contract_num_tokens_from_mult8(int_x, npad)
        int_x = int_x.reshape(bsz, num_concurrent_media, num_chunks, ntok, -1)
        x = torch.cat([x, int_x], dim=-1)
        return x


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.
        Args:
            args (ModelArgs): Model configuration parameters.
        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.
        """
        super().__init__()
        model_parallel_size = fs_init.get_model_parallel_world_size()
        replication_factor = 1
        if model_parallel_size > 8:
            replication_factor = model_parallel_size // MP_SCALE

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_kv_heads *= replication_factor

        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.max_seq_len = args.max_seq_len

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        self.n_heads = args.n_heads

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        if prefix + "wqkv.weight" in state_dict:
            total_n_heads = self.n_heads + self.n_kv_heads * 2
            wqkv = state_dict.pop(prefix + "wqkv.weight")
            head_dim = wqkv.shape[0] // total_n_heads
            dim1 = head_dim * self.n_heads
            dim2 = dim1 + head_dim * self.n_kv_heads
            dim3 = dim1 + head_dim * self.n_kv_heads * 2

            wq = wqkv[:dim1]
            wk = wqkv[dim1:dim2]
            wv = wqkv[dim2:dim3]

            state_dict[prefix + "wq.weight"] = wq
            state_dict[prefix + "wk.weight"] = wk
            state_dict[prefix + "wv.weight"] = wv

    def setup_cache(self, max_batch_size: int, dtype: torch.dtype):
        cache_shape = (
            max_batch_size,
            self.max_seq_len,
            self.n_local_kv_heads,
            self.head_dim,
        )
        device = next(self.parameters()).device
        self.register_buffer(
            "key_cache",
            torch.zeros(
                cache_shape,
                dtype=dtype,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "value_cache",
            torch.zeros(
                cache_shape,
                dtype=dtype,
                device=device,
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        position_ids: torch.LongTensor,
    ):

        xq, xk, xv = [
            F.linear(x, w) for w in [self.wq.weight, self.wk.weight, self.wv.weight]
        ]

        bs, slen, _ = xq.shape

        xq = xq.view(bs, slen, self.n_local_heads, self.head_dim)
        xk = xk.view(bs, xk.shape[1], self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs, xv.shape[1], self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        self.key_cache[:bs, position_ids, ...] = xk
        self.value_cache[:bs, position_ids, ...] = xv

        # TODO: we can avoid slicing on first dimension by always padding to max_batch_size()
        xk = self.key_cache[:bs, ...]
        xv = self.value_cache[:bs, ...]

        xq, xk, xv = [tensor.transpose(1, 2) for tensor in (xq, xk, xv)]

        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)

        attn_output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=mask, dropout_p=0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bs, slen, -1)

        out = F.linear(attn_output, self.wo.weight)
        out = reduce_from_tensor_model_parallel_region(out)
        return out


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.
        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.
        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self._register_load_state_dict_pre_hook(self.load_hook)

    def forward(self, x):
        x1, x3 = [F.linear(x, w) for w in [self.w1.weight, self.w3.weight]]
        x1 = F.silu(x1)
        x_in = x1 * x3
        out = F.linear(x_in, self.w2.weight)
        out = reduce_from_tensor_model_parallel_region(out)
        return out

    def load_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if prefix + "mlp.fc1_weight" in state_dict:
            fc1_weight, fc3_weight = state_dict.pop(prefix + "mlp.fc1_weight").chunk(2)
            state_dict[prefix + "w1.weight"] = fc1_weight
            state_dict[prefix + "w3.weight"] = fc3_weight

        if prefix + "mlp.fc2_weight" in state_dict:
            fc2_weight = state_dict.pop(prefix + "mlp.fc2_weight")
            state_dict[prefix + "w2.weight"] = fc2_weight


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.
        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.
        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.
        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        if prefix + "feed_forward.mlp.layer_norm_weight" in state_dict:
            state_dict[prefix + "ffn_norm.weight"] = state_dict.pop(
                prefix + "feed_forward.mlp.layer_norm_weight"
            )
        if prefix + "attention.wqkv.layer_norm_weight" in state_dict:
            state_dict[prefix + "attention_norm.weight"] = state_dict.pop(
                prefix + "attention.wqkv.layer_norm_weight"
            )

    def setup_cache(self, max_batch_size: int, dtype: torch.dtype):
        self.attention.setup_cache(max_batch_size, dtype)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the TransformerBlock.
        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.
        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.
        """
        h = self.attention.forward(
            x=self.attention_norm(x),
            freqs_cis=freqs_cis,
            mask=mask,
            position_ids=position_ids,
        )
        h = h + x
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class TilePositionEmbedding(nn.Module):
    def __init__(
        self,
        num_tiles: int,
        width: int,
        gated: bool = False,
    ):
        super().__init__()
        self.num_tiles = num_tiles
        self.width = width
        self.embedding = nn.Parameter(
            torch.randn(num_tiles, num_tiles, 1, width) / math.sqrt(width)
        )
        self.gated = gated
        if gated:
            self.gate = nn.Parameter(torch.zeros(1))

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # load the weights from the checkpoint
        embed = state_dict.get(prefix + "embedding")
        if embed is not None:
            # reshape the weights to the correct shape
            nt_old, nt_old, _, w = embed.shape
            logging.info(
                f"Resizing tile embedding from {nt_old}x{nt_old} to {self.num_tiles}x{self.num_tiles}"
            )
            embed_new = TilePositionEmbedding._dynamic_resize(embed, self.num_tiles)
            # assign the weights to the module
            state_dict[prefix + "embedding"] = embed_new

    @staticmethod
    def _dynamic_resize(embed: torch.Tensor, num_tiles: int):
        nt_old, nt_old, _, w = embed.shape
        embed = embed.permute(2, 3, 0, 1)

        embed_new = F.interpolate(
            embed,
            size=(num_tiles, num_tiles),
            mode="bilinear",
            align_corners=True,
        )
        # reshape the weights to the correct shape
        embed_new = embed_new.permute(2, 3, 0, 1)
        return embed_new

    def forward(self, x: torch.Tensor, ar: torch.Tensor, num_tiles: int = None):
        embed = self.embedding
        if num_tiles is None:
            num_tiles = self.num_tiles
        elif num_tiles > self.num_tiles:
            embed = TilePositionEmbedding._dynamic_resize(self.embedding, num_tiles)
        out_pos_embed = torch.zeros(
            x.shape[0], num_tiles, 1, self.width, device=x.device, dtype=x.dtype
        )
        for idx, arx in enumerate(ar):
            w, h = arx
            out_pos_embed[idx, : w * h] = embed[:w, :h].reshape(w * h, 1, self.width)
        if self.gated:
            out_pos_embed = out_pos_embed * self.gate.tanh()
        x = x + out_pos_embed
        return x


def _noinit(x):
    return x


class CrossAttention(torch.nn.Module):
    """Cross attention layer with model-parallel attention layers."""

    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
    ):
        super().__init__()
        self.model_parallel_size = fs_init.get_model_parallel_world_size()
        replication_factor = 1
        if self.model_parallel_size > 8:
            replication_factor = self.model_parallel_size // MP_SCALE
        n_kv_heads *= replication_factor

        assert n_heads % n_kv_heads == 0

        self.wq = ColumnParallelLinear(
            dim,
            n_heads * head_dim,
            bias=False,
            gather_output=False,
            init_method=_noinit,
        )

        self.wk = ColumnParallelLinear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
            gather_output=False,
            init_method=_noinit,
        )
        self.wv = ColumnParallelLinear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
            gather_output=False,
            init_method=_noinit,
        )
        self.wo = RowParallelLinear(
            n_heads * head_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            init_method=_noinit,
        )

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads

        self.q_norm = RMSNorm(
            self.head_dim,
            eps=norm_eps,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            eps=norm_eps,
        )

        # cross-attention heads are model parallel similar to
        # self-attention, and we also use the identical KV head
        # combination to ensure parity with the corresponding
        # trunk LLM (i.e., group query attention) -- @dubeya
        # local heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.n_heads % self.model_parallel_size == 0
        assert self.n_kv_heads % self.model_parallel_size == 0
        self.n_local_heads = self.n_heads // self.model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // self.model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        if prefix + "inner_attention.q_norm.weight" in state_dict:
            q_weight = state_dict.pop(prefix + "inner_attention.q_norm.weight")
            state_dict[prefix + "q_norm.weight"] = q_weight
        if prefix + "inner_attention.k_norm.weight" in state_dict:
            k_weight = state_dict.pop(prefix + "inner_attention.k_norm.weight")
            state_dict[prefix + "k_norm.weight"] = k_weight
        if prefix + "wkv.weight" in state_dict:
            wk, wv = state_dict.pop(prefix + "wkv.weight").chunk(2)
            state_dict[prefix + "wk.weight"] = wk
            state_dict[prefix + "wv.weight"] = wv

    def _compute_xattn_kv_cache(self, xattn_tokens: torch.Tensor) -> torch.Tensor:
        bsz = xattn_tokens.shape[0]
        xk = self.wk(xattn_tokens)
        xv = self.wv(xattn_tokens)

        _, seqlen_y, _ = xk.shape

        xk = xk.view(bsz, seqlen_y, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen_y, self.n_local_kv_heads, self.head_dim)

        xk, xv = [tensor.transpose(1, 2) for tensor in (xk, xv)]

        # repeat k/v heads if n_kv_heads < n_heads
        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)

        xk = self.k_norm(xk)

        return torch.stack([xk, xv])

    def compute_xattn_kv_cache(self, xattn_tokens: torch.Tensor) -> torch.Tensor:
        return self._compute_xattn_kv_cache(xattn_tokens)

    def forward(
        self,
        x: torch.Tensor,
        xattn_mask: torch.Tensor,
        full_text_row_masked_out_mask: torch.Tensor,
        xattn_cache: torch.Tensor,
    ) -> torch.Tensor:
        xq = F.linear(x, self.wq.weight)
        bsz, seqlen, _ = x.shape

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq = self.q_norm(xq)
        xq = xq.transpose(1, 2)

        xk, xv = xattn_cache

        output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=xattn_mask, dropout_p=0.0
        )
        output = output * full_text_row_masked_out_mask
        output = output.transpose(1, 2).contiguous().reshape(bsz, seqlen, -1)

        out = F.linear(output, self.wo.weight)
        out = reduce_from_tensor_model_parallel_region(out)
        return out


class CrossAttentionTransformerBlock(torch.nn.Module):
    """Cross-attention transformer block with tanh-gated attention and feedforward."""

    def __init__(
        self,
        args: ModelArgs,
        layer_id: int,
        no_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = CrossAttention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            norm_eps=args.norm_eps,
        )

        self.attention_norm = RMSNorm(
            args.dim,
            eps=args.norm_eps,
        )
        self.gate_attn = torch.nn.Parameter(torch.zeros(1))

        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            multiple_of=args.multiple_of,
        )
        self.ffn_norm = RMSNorm(
            args.dim,
            eps=args.norm_eps,
        )
        self.gate_ffwd = torch.nn.Parameter(torch.zeros(1))

        self._register_load_state_dict_pre_hook(self.load_hook)
        self.no_ffn = no_ffn

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        if prefix + "gate_attn" in state_dict:
            attn_gate = state_dict.pop(prefix + "gate_attn")
            if attn_gate.dim() == 1:
                attn_gate = attn_gate[0].view(1)
            if attn_gate.dim() == 3:
                attn_gate = attn_gate.view(1)
            state_dict[prefix + "gate_attn"] = attn_gate
        if prefix + "gate_ffwd" in state_dict:
            ffn_gate = state_dict.pop(prefix + "gate_ffwd")
            if ffn_gate.dim() == 1:
                ffn_gate = ffn_gate[0].view(1)
            if ffn_gate.dim() == 3:
                ffn_gate = ffn_gate.view(1)
            state_dict[prefix + "gate_ffwd"] = ffn_gate
        if prefix + "feed_forward.mlp.layer_norm_weight" in state_dict:
            state_dict[prefix + "ffn_norm.weight"] = state_dict.pop(
                prefix + "feed_forward.mlp.layer_norm_weight"
            )
        if prefix + "attention.wq.layer_norm_weight" in state_dict:
            state_dict[prefix + "attention_norm.weight"] = state_dict.pop(
                prefix + "attention.wq.layer_norm_weight"
            )

    def compute_xattn_kv_cache(self, xattn_tokens: torch.Tensor) -> torch.Tensor:
        return self.attention.compute_xattn_kv_cache(xattn_tokens)

    def forward(
        self,
        x: torch.Tensor,
        xattn_mask: torch.Tensor,
        full_text_row_masked_out_mask: Tuple[torch.Tensor, torch.Tensor],
        xattn_cache: torch.Tensor,
    ) -> torch.Tensor:
        _attn_out = self.attention(
            x=self.attention_norm(x),
            xattn_mask=xattn_mask,
            xattn_cache=xattn_cache,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
        )
        h = x + self.gate_attn.tanh() * _attn_out
        _ffn = self.feed_forward(self.ffn_norm(h))
        _ffn = full_text_row_masked_out_mask[:, 0] * _ffn  # type: ignore
        h = h + self.gate_ffwd.tanh() * _ffn * float(not self.no_ffn)
        return h


class DummyCrossAttentionTransformerBlock:
    """Dummy cross-attention transformer block with tanh-gated attention and feedforward."""

    def __call__(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return x


class DummySelfAttentionTransformerBlock:
    """Dummy self-attention transformer block"""

    def __call__(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return x


def _stack_images(
    images: List[List[PIL_Image.Image]],
    max_num_chunks: int,
    image_res: int,
    max_num_images: int,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Takes a list of list of images and stacks them into a tensor.
    This function is needed since images can be of completely
    different resolutions and aspect ratios.
    """
    out_images, out_num_chunks = [], []
    for imgs_sample in images:
        out_images_i = torch.zeros(
            max_num_images,
            max_num_chunks,
            3,
            image_res,
            image_res,
        )
        _num_chunks = []
        for j, chunks_image in enumerate(imgs_sample):
            out_images_i[j, : chunks_image.shape[0]] = chunks_image
            _num_chunks.append(chunks_image.shape[0])
        out_images.append(out_images_i)
        out_num_chunks.append(_num_chunks)
    return torch.stack(out_images), out_num_chunks


def _pad_masks(
    all_masks: List[List[List[int]]],
    all_num_chunks: List[List[int]],
    total_len: int,
    max_num_chunks: int,
) -> torch.Tensor:
    dtype = torch.bfloat16
    inf_value = get_negative_inf_value(dtype)

    bsz = len(all_masks)
    max_num_media = max([len(m) for m in all_masks])

    out_masks = torch.full(
        (bsz, total_len, max_num_media, max_num_chunks),
        inf_value,
        dtype=dtype,
    )

    for idx, (mask, num_chunks) in enumerate(zip(all_masks, all_num_chunks)):
        for mask_idx, (mask_elem, mask_num_chunks) in enumerate(zip(mask, num_chunks)):
            if len(mask_elem) == 2:
                mask_elem[1] = min(mask_elem[1], total_len)
                if mask_elem[1] == -1:
                    mask_elem[1] = total_len
                out_masks[
                    idx, mask_elem[0] : mask_elem[1], mask_idx, :mask_num_chunks
                ].fill_(0.0)

    return out_masks
