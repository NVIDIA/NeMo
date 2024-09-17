# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.
import copy
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image as PIL_Image
from megatron.core import tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    make_viewless_tensor,
)
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


class ColumnParallelConv2dPatch(MegatronModule):
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
            config: TransformerConfig,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]],
            bias: Optional[bool] = False,
    ) -> None:
        super().__init__(config=config)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self._unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride)
        self._linear = TEColumnParallelLinear(
            in_channels * kernel_size[0] * kernel_size[1],
            out_channels,
            bias=bias,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='conv1',
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._unfold(x)
        x = x.permute(0, 2, 1)
        x = F.linear(x, self._linear.weight)
        x = tensor_parallel.gather_from_tensor_model_parallel_region(x)
        return x


from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from megatron.core.transformer.transformer_block import TransformerBlock


# Use this spec for an implementation using modules in TE
def get_image_transformer_layer_spec() -> ModuleSpec:
    image_transformer_submodules = TransformerLayerSubmodules(
        input_layernorm=TENorm,
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.padding},
            submodules=SelfAttentionSubmodules(
                linear_qkv=TEColumnParallelLinear,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
                q_layernorm=IdentityOp,
                k_layernorm=IdentityOp,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=TENorm,
        mlp=ModuleSpec(
            module=MLP, submodules=MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear, ),
        ),
        mlp_bda=get_bias_dropout_add,
    )
    return ModuleSpec(module=ImageTransformerLayer, submodules=image_transformer_submodules)


class ImageTransformerLayer(TransformerLayer):
    def __init__(
            self,
            config: TransformerConfig,
            submodules: TransformerLayerSubmodules,
            layer_number: int = 1,
            hidden_dropout: float = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
        )
        self.gated = self.config.gated
        if self.gated:
            self.gate_attn = nn.Parameter(torch.zeros(1))
            self.gate_ffn = nn.Parameter(torch.zeros(1))

    def forward(
            self,
            hidden_states,
            attention_mask,
            context=None,
            context_mask=None,
            rotary_pos_emb=None,
            inference_params=None,
            packed_seq_params=None,
    ):
        if len(hidden_states.shape) == 4:
            intermediate_output = hidden_states[:-1]
            hidden_states = hidden_states[-1]
        else:
            intermediate_output = torch.empty((0, *hidden_states.shape), dtype=hidden_states.dtype,
                                              device=hidden_states.device)

        # hidden_states: [s, b, h]

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )
        _gate_attn = 1 if not self.gated else self.gate_attn.tanh()
        attention_output_with_bias = _gate_attn * attention_output_with_bias

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_params=inference_params,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        _gate_ffn = 1 if not self.gated else self.gate_ffn.tanh()
        mlp_output_with_bias = _gate_ffn * mlp_output_with_bias

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        output = output.unsqueeze(0)
        output = torch.cat((intermediate_output, output, output), dim=0)

        return output, context


class VisionEncoder(MegatronModule):
    def __init__(
            self,
            config: TransformerConfig,
            max_num_tiles: int,
            image_size: int = 224,
            patch_size: int = 14,
            in_channels: int = 3,
            pre_process: bool = True,
            post_process: bool = True,
            return_intermediate=None,
    ):
        super().__init__(config=config)
        self.return_intermediate = return_intermediate
        self.max_num_tiles = max_num_tiles
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        self.pre_process = pre_process
        self.post_process = post_process

        width = config.hidden_size
        self.conv1 = ColumnParallelConv2dPatch(
            config=config,
            in_channels=in_channels,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width)
        )
        self.ln_post = LayerNorm(width)
        self.ln_pre = LayerNorm(width)
        self.transformer = TransformerBlock(
            config=self.config,
            spec=get_image_transformer_layer_spec(),
            post_layer_norm=False,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )
        # pre and post tile position embedding
        global_config = copy.deepcopy(self.config)
        global_config.num_layers = self.config.num_global_layers
        self.global_transformer = TransformerBlock(
            config=global_config,
            spec=get_image_transformer_layer_spec(),
            post_layer_norm=False,
            pre_process=self.pre_process,
            post_process=self.post_process,
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


class TilePositionEmbedding(torch.nn.Module):
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
                idx, mask_elem[0]: mask_elem[1], mask_idx, :mask_num_chunks
                ].fill_(0.0)

    return out_masks
