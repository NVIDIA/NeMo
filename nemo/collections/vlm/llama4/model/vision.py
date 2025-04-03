# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Union

import torch
from torch import nn
from torch import einsum

from megatron.core.transformer.module import MegatronModule

from nemo.collections.vlm import MultimodalProjectorConfig
from nemo.collections.vlm.mllama.model.vision import ColumnParallelConv2dPatch

try:
    from megatron.core.extensions.transformer_engine import (
        TENorm,
    )

    NORM_IMPL = TENorm
except ImportError:
    from nemo.utils import logging

    # These Defaults are needed to make sure the code compiles
    TENorm = None
    NORM_IMPL = torch.nn.LayerNorm

    logging.warning(
        "Failed to import Transformer Engine dependencies. "
        "`from megatron.core.extensions.transformer_engine import *`"
        "If using NeMo Run, this is expected. Otherwise, please verify the Transformer Engine installation."
    )
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.enums import ModelType
from megatron.core.models.common.vision_module.vision_module import VisionModule

from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock

from nemo.collections.vlm.vision.base import CLIPViTConfig


@dataclass
class Llama4VisionConfig(CLIPViTConfig):
    """Intern ViT Base Config"""

    vision_model_type: str = "llama4"
    patch_dim: int = 14
    img_h: int = 336
    img_w: int = 336
    num_layers: int = 34
    num_attention_heads: int = 16
    num_query_groups: int = 16
    kv_channels: int = 88
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    hidden_size: int = 1408
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    ffn_hidden_size: int = 5632
    output_dim: int = 4096
    gated_linear_unit: bool = False
    activation_func: Callable = torch.nn.functional.gelu
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization: str = 'LayerNorm'
    layernorm_epsilon: float = 1e-6
    apply_rope_fusion: bool = False
    pixel_shuffle_ratio: float = 0.5
    rotary_interleaved: bool = True
    transformer_layer_spec: Optional[ModuleSpec] = None

    def configure_model(self) -> "Llama4ViTModel":
        # pylint: disable=C0115,C0116
        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            from nemo.collections.vlm.layer_specs import get_layer_spec_te
            transformer_layer_spec = get_layer_spec_te(is_vit=True)

        return Llama4ViTModel(
            self,
            transformer_layer_spec,
            ln_pre_impl=self.ln_pre_impl,
            ln_post_impl=self.ln_post_impl,
            add_class_token=self.add_class_token,
            class_token_len=self.class_token_len,
            patch_dim=self.patch_dim,
            img_h=self.img_h,
            img_w=self.img_w,
            model_subtype=self.vision_model_type,
        )


class PackingIndex:
    Z = 0  # Z (time) coordinate of the token in the original sample
    Y = 1  # Y (height) coordinate of the token in the original sample
    X = 2  # X (width) coordinate of the token in the original sample
    TIME = 3  # Total number of time units (frames) in the original sample
    HEIGHT = 4  # Height of the original sample
    WIDTH = 5  # Width of the original sample
    # USE INDEX TO CHECK THE TYPE OF THE TOKEN (see ID fields below)
    IDX = 6  # Full index of the token in the original sample (x + y * w + z * w * h)
    BATCH_IDX = 7  # Which batch element this token belongs to. Note the batch idx of padding tokens is BATCH_SIZE

    # Total size of the enum, remember to update this!
    NUM_METADATA = 8

    # Note: For padding tokens IDX = -1
    #       For cls tokens,    IDX = -2
    ID_CLS_TOKEN = -2
    ID_PAD_TOKEN = -1


class PixelShuffle(nn.Module):
    def __init__(self, ps_ratio):
        super().__init__()
        self.ps_ratio = ps_ratio

    def forward(self, x):
        # x: [B, N, C], N = number of patches
        assert self.ps_ratio is not None, "ps_ratio is required for pixel shuffle"
        assert x.dim() == 3, "pixel shuffle requires encoded patches [B, N, C]"
        hh = ww = int(math.sqrt(x.shape[1]))
        x = x.reshape(x.shape[0], hh, ww, -1)
        x = pixel_shuffle_op(x, ps_ratio=self.ps_ratio)
        pixel_shuffle_patches = x.reshape(x.shape[0], -1, x.shape[-1])
        return pixel_shuffle_patches


def pixel_shuffle_op(input_x, ps_ratio):
    n, w, h, c = input_x.size()
    input_x = input_x.view(n, w, int(h * ps_ratio), int(c / ps_ratio))
    input_x = input_x.permute(0, 2, 1, 3).contiguous()
    input_x = input_x.view(
        n,
        int(h * ps_ratio),
        int(w * ps_ratio),
        int(c / (ps_ratio * ps_ratio)),
    )
    input_x = input_x.permute(0, 2, 1, 3).contiguous()
    return input_x

class PixelShuffleMLP(MegatronModule):
    def __init__(
            self,
            config: TransformerConfig,
            ps_ratio: float,
            input_dim: int,
            output_dim: int = 4096,
            add_fc: bool = False,
    ):
        super().__init__(config)
        self.pixel_shuffle = PixelShuffle(ps_ratio)
        self.mlp_config = MultimodalProjectorConfig(
            projector_type="mcore_mlp",
            input_size=int(input_dim // (ps_ratio**2)),
            hidden_size=output_dim,
            ffn_hidden_size=output_dim,
            bias=False,
            bias_activation_fusion=False,
        )
        self.mlp = self.mlp_config.configure_model()
        if add_fc:
            raise NotImplementedError

    def forward(self, encoded_patches: torch.Tensor) -> torch.Tensor:
        encoded_patches = self.pixel_shuffle(encoded_patches)
        activation_func = self.mlp.encoder.activation_func
        return activation_func(self.mlp(encoded_patches))


class Llama4ViTModel(VisionModule):
    """CLIP ViT vision model.

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
        ln_pre_impl (ModuleSpec or type): Specifies the layer norm type to use for ln_pre.
        add_class_token (bool, optional): Include a class token. Defaults to True.
        class_token_len (int): Class token length. Defaults to 1 but 8 may be faster.
        patch_dim (int): Image patch size.
        img_h (int): Input image height.
        img_w (int): Input image width.
    """

    def __init__(
            self,
            transformer_config: TransformerConfig,
            transformer_layer_spec: ModuleSpec,
            ln_pre_impl: Union[ModuleSpec, type] = NORM_IMPL,
            ln_post_impl: Union[ModuleSpec, type] = NORM_IMPL,
            add_class_token: bool = True,
            class_token_len: int = 1,
            patch_dim: int = 14,
            img_h: int = 336,
            img_w: int = 336,
            model_subtype: str = "llama4",
    ) -> None:

        super().__init__(config=transformer_config)

        self.class_token_len = class_token_len
        self.visual_hidden_size = transformer_config.hidden_size
        self.patch_dim = patch_dim
        self.img_h = img_h
        self.img_w = img_w

        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w

        self.add_class_token = add_class_token
        self.class_token_len = class_token_len

        self.seq_length = self.num_patches + (self.class_token_len if self.add_class_token else 0)

        self.ln_pre = build_module(
            ln_pre_impl,
            config=transformer_config,
            hidden_size=self.visual_hidden_size,
            eps=transformer_config.layernorm_epsilon,
        )
        self.ln_post = build_module(
            ln_post_impl,
            config=transformer_config,
            hidden_size=self.visual_hidden_size,
            eps=transformer_config.layernorm_epsilon,
        )

        self.conv1 = ColumnParallelConv2dPatch(
            config=transformer_config,
            in_channels=3,
            out_channels=self.visual_hidden_size,
            kernel_size=self.patch_dim,
            stride=self.patch_dim,
            bias=False,
        )

        self.position_ids = torch.arange(self.seq_length).expand(1, -1).cuda()

        self.position_embeddings = torch.nn.Embedding(self.seq_length, self.visual_hidden_size)

        self.add_class_token = add_class_token
        if self.add_class_token:
            scale = self.visual_hidden_size ** -0.5
            self.class_token = torch.nn.Parameter(
                scale * torch.randn(1, self.class_token_len, self.visual_hidden_size)
            )

        self.model_type = ModelType.encoder_or_decoder

        # Transformer layers.
        # TODO: Make pre_process and post_process configurable.
        # NOTE: a final layer norm and/or linear layer in some implementations are omitted here.
        # They can be added separately where needed.
        self.decoder = TransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False,
        )
        self.adapter = PixelShuffleMLP(
            config=transformer_config,
            ps_ratio=transformer_config.pixel_shuffle_ratio,
            input_dim=transformer_config.hidden_size,
            output_dim=transformer_config.output_dim,
        )

        self.output_dim = transformer_config.output_dim
        self.rotary_pos_emb = self.get_rope_emb()

    def get_rope_emb(self):
        # Adapted from Llama4
        patch_h = patch_w = self.patch_dim
        idx_h, idx_w = self.img_h // patch_h, self.img_w // patch_w
        img_idx = torch.arange(self.img_h * self.img_w // (patch_h * patch_w), dtype=torch.int32)
        img_idx = img_idx.reshape(idx_h * idx_w, 1)
        img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
        img_idx[-1, -1] = PackingIndex.ID_CLS_TOKEN

        packed_img_idx = torch.empty(
            img_idx.shape[0],
            img_idx.shape[1],
            PackingIndex.NUM_METADATA - 1,
            dtype=torch.int32,
        )
        packed_img_idx[:, :, PackingIndex.Y] = img_idx // idx_w
        packed_img_idx[:, :, PackingIndex.X] = img_idx % idx_w
        packed_img_idx[:, :, PackingIndex.HEIGHT].fill_(idx_h)
        packed_img_idx[:, :, PackingIndex.WIDTH].fill_(idx_w)
        packed_img_idx[:, :, PackingIndex.IDX] = img_idx
        packed_img_idx = packed_img_idx.reshape(1, -1, PackingIndex.NUM_METADATA - 1)
        self.packed_img_idx = packed_img_idx  # for positional embedding load hook

        # compute rope freqs
        rope_freq = self.get_rope_freqs(
            self.config.hidden_size // self.config.num_attention_heads // 2
        )
        freqs_x = self.compute_rope_freqs(rope_freq, packed_img_idx[:, :, PackingIndex.X] + 1)
        freqs_y = self.compute_rope_freqs(rope_freq, packed_img_idx[:, :, PackingIndex.Y] + 1)
        freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
        # disable RoPE for padding and cls tokens
        freqs = freqs.masked_fill(packed_img_idx[:, :, PackingIndex.IDX, None] < 0, 0)
        freqs = freqs.squeeze(0)
        rotary_pos_emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
            freqs.shape[0], -1
        )
        # emb [seq_length, .., dim]
        rotary_pos_emb = rotary_pos_emb[:, None, None, :]
        return rotary_pos_emb.cuda()

    def get_rope_freqs(self, dim, theta=10000):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        return freqs

    @torch.amp.autocast("cuda", enabled=False)
    def compute_rope_freqs(self, freqs, t):
        freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = freqs.repeat_interleave(2, dim=-1)
        return freqs

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.decoder.set_input_tensor(input_tensor)

    def _get_empty_sequence(self, h):
        return torch.zeros(
            h.shape[0],
            h.shape[1],
            self.output_dim,
            device=h.device,
            dtype=h.dtype,
        )

    def _encode(
            self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function of the CLIP ViT Model. This function passes the input tensors
        through the embedding layer and then the transformer.

        Args:
            x (torch.Tensor): input data of shape [batch, img_h, img_w]
            attention_mask (torch.Tensor with dtype=bool): Attention mask to use.

        Returns:
            x (torch.Tensor): output after final transformer block of shape [b, s, h].
        """

        # NOTE(from Llama4 reference): in Llama4 bsz=bsz*num_tiles, num_chunks=1
        # TODO(yuya): clean up logic for dims
        if x.ndim == 5:
            num_concurrent_media = 1
            bsz, num_chunks, nch, h, w = x.shape
        else:
            bsz, num_concurrent_media, num_chunks, nch, h, w = x.shape
        x = x.reshape(bsz * num_concurrent_media * num_chunks, nch, h, w)

        x = self.conv1(x)  # [batch, grid ** 2, hidden_size]

        if self.add_class_token:
            class_token = self.class_token.expand(
                x.shape[0], -1, -1
            )  # [batch, class_token_len, hidden_size]
            x = torch.cat(
                [x, class_token], dim=1
            )  # [batch, grid ** 2 + class_token_len, hidden_size]

        assert x.shape[1] == self.seq_length, f"{x.shape[1]} != {self.seq_length}"
        x = x + self.position_embeddings(self.position_ids)
        if self.ln_pre:
            x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # [b, s, h] -> [s, b, h]
        # `permute` can make the tensor non-contiguous, breaking pipelining.
        x = x.contiguous()

        x = self.decoder(
            x, attention_mask,
            rotary_pos_emb=self.rotary_pos_emb,
        )
        x = x.permute(1, 0, 2)  # [s, b, h] -> [b, s, h]
        x = x.contiguous()
        if self.ln_post:
            x = self.ln_post(x)

        return x

    def forward(
            self,
            image_batch: List[List[torch.Tensor]],
            image_mask: torch.Tensor,
            h_ref: torch.Tensor,
    ) -> torch.Tensor:
        # TODO(yuya): Move input processing and output processing to base model
        # to keep vit submodule clean

        images_flattened = [image for sample in image_batch for image in sample]
        images_flattened = torch.vstack(images_flattened).unsqueeze(1)
        embedding = self._encode(images_flattened)
        # remove cls token output
        embedding = embedding[:, :-1, :]
        projected_embedding = self.adapter(embedding)

        h_image = self._get_empty_sequence(h_ref)
        return scatter_embeddings(image_batch, image_mask, h_image, projected_embedding)


def scatter_embeddings(image_batch, image_mask, h_image, encoded_patches_proj):
    # If dynamic transform is used and the batch contains 2 images (where image_1 has 2 chunks and image_2 has 3 chunks),
    # `num_images_per_sequence` now records the number of chunks per image as `[2, 3]`.
    # `encoded_patches_proj.split` will then split the image chunks into 2 groups: `[image_1_chunks, image_2_chunks]`.
    num_images_per_sequence = [sum(image.size(0) for image in sample_images) for sample_images in image_batch]

    assert not torch.isnan(encoded_patches_proj).any()
    assert sum(num_images_per_sequence) == encoded_patches_proj.size(0), (
        f"{sum(num_images_per_sequence)=} != {encoded_patches_proj.shape=}"
    )

    encoded_patches_list = encoded_patches_proj.split(num_images_per_sequence, dim=0)
    for index in range(h_image.size(0)):
        encoded_patches_per_sample = encoded_patches_list[index]
        sample_image_mask = image_mask[index]

        if encoded_patches_per_sample.numel() == 0:
            continue
        encoded_patches_per_sample = encoded_patches_per_sample.contiguous().view(
            -1, encoded_patches_per_sample.size(-1)
        )

        n_tokens_to_fill = sample_image_mask.sum()
        assert n_tokens_to_fill <= encoded_patches_per_sample.size(0)

        h_image[index].masked_scatter_(
            sample_image_mask.expand(-1, h_image.size(-1)),
            encoded_patches_per_sample[:n_tokens_to_fill],
        )

    return h_image
