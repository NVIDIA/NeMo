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

from typing import Optional

import torch
import torch.nn.functional as F

from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig


class VisionRotaryEmbedding(torch.nn.Module):
    # pylint: disable=C0115,C0116

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        # pylint: disable=C0115,C0116
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen25VisionModel(VisionModule):
    """Qwen2.5-VL vision model."""

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        add_class_token: bool = False,
        class_token_len: int = 1,
        patch_dim: int = 14,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        spatial_patch_size: int = 14,
        img_h: int = 336,
        img_w: int = 336,
        fullatt_block_indexes: list[int] = [7, 15, 23, 31],
        window_size: int = 112,
    ) -> None:

        super().__init__(config=transformer_config)

        if has_config_logger_enabled(transformer_config):
            log_config_to_disk(transformer_config, locals(), prefix=type(self).__name__)

        self.class_token_len = class_token_len
        self.visual_hidden_size = transformer_config.embed_dim
        self.patch_dim = patch_dim
        self.temporal_patch_size = temporal_patch_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.window_size = window_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_patch_size = spatial_patch_size
        self.merge_hidden_size = self.visual_hidden_size * (spatial_merge_size**2)
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.img_h = img_h
        self.img_w = img_w
        self.in_channels = 3

        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w

        self.add_class_token = add_class_token
        self.class_token_len = class_token_len

        self.seq_length = self.num_patches + (self.class_token_len if self.add_class_token else 0)

        kernel_size = [temporal_patch_size, patch_dim, patch_dim]
        self.conv1 = torch.nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.visual_hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

        head_dim = transformer_config.embed_dim // transformer_config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.add_class_token = add_class_token
        if self.add_class_token:
            self.class_token = torch.nn.Parameter(torch.randn(1, self.class_token_len, self.visual_hidden_size))

        self.model_type = ModelType.encoder_or_decoder

        # Transformer layers.
        # TODO: Make pre_process and post_process configurable.
        # NOTE: a final layer norm and/or linear layer in some implementations are omitted here.
        # They can be added separately where needed.
        self.decoder = TransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=True,
        )

        self.window_index = None

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.decoder.set_input_tensor(input_tensor)

    def rot_pos_emb(self, grid_thw):
        # pylint: disable=C0115,C0116
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_packed_seq_params(
        self,
        grid_thw: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        # pylint: disable=C0115,C0116
        from megatron.core.packed_seq_params import PackedSeqParams

        if grid_thw is not None:
            seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
            cu_seqlens = seqlens.cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
            cu_seqlens = cu_seqlens.squeeze()
        else:
            cu_seqlens = cu_seqlens.squeeze()
            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

        max_seqlen = seqlens.max().item()

        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format='thd',
        )

    def get_window_index(self, grid_thw):
        # pylint: disable=C0115,C0116
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_dim

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(
        self, x: torch.Tensor, grid_thw: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function of the Qwen2.5 Vision Model. This function passes the input tensors
        through the embedding layer and then the transformer.

        Args:
            x (torch.Tensor): input data of shape [batch, img_h, img_w]
            grid_thw (torch.Tensor): The temporal, height and width of feature shape of each image/frame.
            attention_mask (torch.Tensor with dtype=bool): Attention mask to use.

        Returns:
            x (torch.Tensor): output after final transformer block.
        """
        # pylint: disable=C0301
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_dim, self.patch_dim)
        x = self.conv1(x).view(-1, self.visual_hidden_size)  # [seqlen, hidden_size]
        # add batch dim

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=x.device,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = x.size()
        # Refer to https://github.com/huggingface/transformers/blob/be37d34f44ff1bc928e59ffb8a30adecab8835a8/src/
        # transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L527C9-L530C59
        x = x.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        x = x[window_index, :, :]
        x = x.reshape(seq_len, -1)
        x = x.unsqueeze(1)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        # from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/common/embeddings/rotary_pos_embedding.py#L158
        rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/common/embeddings/rotary_pos_embedding.py#L164
        rotary_pos_emb = rotary_pos_emb[:, None, None, :]

        packed_seq_params = self.get_packed_seq_params(grid_thw)
        packed_seq_params_full = self.get_packed_seq_params(None, cu_window_seqlens)

        x = self.decoder(
            x,
            attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            packed_seq_params_full=packed_seq_params_full,
            fullatt_block_indexes=self.fullatt_block_indexes,
        )
        x = x.squeeze(1).view(-1, self.merge_hidden_size)
        self.window_index = window_index
        return x
