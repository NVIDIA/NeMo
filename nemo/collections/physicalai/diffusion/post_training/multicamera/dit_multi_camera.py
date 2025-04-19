# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# pylint: disable=C0115,C0116,C0301

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import attrs
import einops
import numpy as np
import torch
import wandb
from einops import rearrange, repeat
from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.packed_seq_params import PackedSeqParams
from torch import Tensor, nn
from torch.distributed import ProcessGroup, get_process_group_ranks
from torchvision import transforms
from typing_extensions import override

from nemo.collections.diffusion.models.dit.dit_model_7b import (
    DiTCrossAttentionModel7B,
    PatchEmbed,
    cat_outputs_cp,
    get_1d_sincos_pos_embed_from_grid,
    split_inputs_cp,
)
from nemo.collections.diffusion.models.model import DiT7BConfig, DiTModel, dynamic_import
from nemo.collections.diffusion.sampler.conditioner import (
    AbstractEmbModel,
    DataType,
    Edify4Condition,
    TrainingOnlyEmbModel,
    VideoConditioner,
)
from nemo.collections.diffusion.sampler.conditioner_configs import (
    FPSConfig,
    ImageSizeConfig,
    NumFramesConfig,
    PaddingMaskConfig,
    TextConfig,
)
from nemo.collections.diffusion.sampler.cosmos.cosmos_diffusion_pipeline import CosmosDiffusionPipeline
from nemo.collections.diffusion.sampler.res.res_sampler import COMMON_SOLVER_OPTIONS


class MultiCameraVideoPositionEmb(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.cp_group = None

    def enable_context_parallel(self, cp_group: ProcessGroup):
        self.cp_group = cp_group

    def disable_context_parallel(self):
        self.cp_group = None

    def forward(self, x_B_T_H_W_C: torch.Tensor, fps=Optional[torch.Tensor]) -> torch.Tensor:
        """
        With CP, the function assume that the input tensor is already split. It delegates the embedding generation to generate_embeddings function.
        """
        B_T_H_W_C = x_B_T_H_W_C.shape
        if self.cp_group is not None:
            cp_ranks = get_process_group_ranks(self.cp_group)
            cp_size = len(cp_ranks)
            B, T, H, W, C = B_T_H_W_C
            B_T_H_W_C = (B, T * cp_size, H, W, C)
        embeddings = self.generate_embeddings(B_T_H_W_C, fps=fps)

        if self.cp_group is not None:
            if isinstance(self, MultiCameraVideoRopePosition3DEmb):
                seq_dim = 1
                embeddings = rearrange(embeddings, "(V T) H W D -> V (T H W) 1 1 D", V=self.n_cameras).float()
                # rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()
                embeddings = split_inputs_cp(x=embeddings, seq_dim=seq_dim, cp_group=self.cp_group)
                embeddings = rearrange(embeddings, "V T 1 1 D -> (V T) 1 1 D", V=self.n_cameras).float()
            else:
                seq_dim = 1
                embeddings = rearrange(embeddings, "B (V T) H W C -> (B V) T H W C", V=self.n_cameras)
                embeddings = split_inputs_cp(x=embeddings, seq_dim=seq_dim, cp_group=self.cp_group)
                embeddings = rearrange(embeddings, "(B V) T H W C -> B (V T) H W C", V=self.n_cameras)
        else:
            if isinstance(self, MultiCameraVideoRopePosition3DEmb):
                embeddings = rearrange(embeddings, "t h w d -> (t h w) 1 1 d").float()

        return embeddings

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor]):
        raise NotImplementedError


class MultiCameraVideoRopePosition3DEmb(MultiCameraVideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        n_cameras: int = 4,
        **kwargs,  # used for compatibility with other positional embeddings; unused in this class
    ):
        del kwargs
        super().__init__()
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float))
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.n_cameras = n_cameras
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().cuda() / dim_h,
            persistent=False,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().cuda() / dim_t,
            persistent=False,
        )

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

    def generate_embedding_for_batch(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ):
        """
        Generate embeddings for the given input size.

        Args:
            B_T_H_W_C (torch.Size): Input tensor size (Batch, Time, Height, Width, Channels).
            fps (Optional[torch.Tensor], optional): Frames per second. Defaults to None.
            h_ntk_factor (Optional[float], optional): Height NTK factor. If None, uses self.h_ntk_factor. Defaults to None.
            w_ntk_factor (Optional[float], optional): Width NTK factor. If None, uses self.w_ntk_factor. Defaults to None.
            t_ntk_factor (Optional[float], optional): Time NTK factor. If None, uses self.t_ntk_factor. Defaults to None.

        Returns:
            Not specified in the original code snippet.
        """
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        B, T, H, W, _ = B_T_H_W_C
        uniform_fps = (fps is None) or (fps.min() == fps.max())
        assert uniform_fps  # only support uniform fps now

        assert (
            uniform_fps or B == 1 or T == 1
        ), "For video batch, batch size should be 1 for non-uniform fps. For image batch, T should be 1"
        assert (
            H <= self.max_h and W <= self.max_w
        ), f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w}) configured for positional embedding. Please adjust the input size or increase the maximum dimensions in the model configuration."
        half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)

        # apply sequence scaling in temporal dimension
        if fps is None:  # image case
            assert T == 1, "T should be 1 for image batch."
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
        else:
            half_emb_t = torch.outer(self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        return em_T_H_W_D

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ):
        """
        Generate embeddings for the given input size. The camera view dimension is merged in the T dimension

        Args:
            B_T_H_W_C (torch.Size): Input tensor size (Batch, Time * Views, Height, Width, Channels).
            fps (Optional[torch.Tensor], optional): Frames per second. Defaults to None.
            h_ntk_factor (Optional[float], optional): Height NTK factor. If None, uses self.h_ntk_factor. Defaults to None.
            w_ntk_factor (Optional[float], optional): Width NTK factor. If None, uses self.w_ntk_factor. Defaults to None.
            t_ntk_factor (Optional[float], optional): Time NTK factor. If None, uses self.t_ntk_factor. Defaults to None.

        Returns:
            Not specified in the original code snippet.
        """

        B, T, H, W, C = B_T_H_W_C

        single_camera_B_T_H_W_C = (B, T // self.n_cameras, H, W, C)
        em_T_H_W_D = torch.cat(
            [
                self.generate_embedding_for_batch(
                    single_camera_B_T_H_W_C,
                    fps=fps,
                    h_ntk_factor=h_ntk_factor,
                    w_ntk_factor=w_ntk_factor,
                    t_ntk_factor=t_ntk_factor,
                )
                for item in range(self.n_cameras)
            ],
            dim=0,
        )

        return em_T_H_W_D
        # return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()


class MultiCameraSinCosPosEmbAxis(MultiCameraVideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        interpolation: str,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        n_cameras: int = 4,
        **kwargs,
    ):
        # TODO: (qsh 2024-11-08) add more interpolation methods and args for extrapolation fine-tuning
        """
        Args:
            interpolation (str): we curretly only support "crop", ideally when we need extrapolation capacity, we should adjust frequency or other more advanced methods. they are not implemented yet.
        """
        del kwargs  # unused
        self.n_cameras = n_cameras
        super().__init__()
        self.interpolation = interpolation
        assert self.interpolation in ["crop"], f"Unknown interpolation method {self.interpolation}"

        dim = model_channels
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"

        # rescale pos id is equivalent to rescale frequency
        emb_h = get_1d_sincos_pos_embed_from_grid(dim_h, pos=np.arange(len_h) * 1.0 / h_extrapolation_ratio)
        emb_w = get_1d_sincos_pos_embed_from_grid(dim_w, pos=np.arange(len_w) * 1.0 / w_extrapolation_ratio)
        emb_t = get_1d_sincos_pos_embed_from_grid(dim_t, pos=np.arange(len_t) * 1.0 / t_extrapolation_ratio)

        self.register_buffer("pos_emb_h", torch.from_numpy(emb_h).float(), persistent=False)
        self.register_buffer("pos_emb_w", torch.from_numpy(emb_w).float(), persistent=False)
        self.register_buffer("pos_emb_t", torch.from_numpy(emb_t).float(), persistent=False)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor]) -> torch.Tensor:
        B, T, H, W, C = B_T_H_W_C

        single_camera_T = T // self.n_cameras

        if self.interpolation == "crop":
            emb_h_H = self.pos_emb_h[:H]
            emb_w_W = self.pos_emb_w[:W]
            emb_t_T = self.pos_emb_t[:single_camera_T]
            emb = torch.cat(
                [
                    torch.cat(
                        [
                            repeat(emb_t_T, "t d-> b t h w d", b=B, h=H, w=W),
                            repeat(emb_h_H, "h d-> b t h w d", b=B, t=single_camera_T, w=W),
                            repeat(emb_w_W, "w d-> b t h w d", b=B, t=single_camera_T, h=H),
                        ],
                        dim=-1,
                    )
                    for _ in range(self.n_cameras)
                ],
                1,
            )
            assert list(emb.shape)[:4] == [B, T, H, W], f"bad shape: {list(emb.shape)[:4]} != {B, T, H, W}"
            return emb

        raise ValueError(f"Unknown interpolation method {self.interpolation}")


class MultiCameraDiTCrossAttentionModel7B(DiTCrossAttentionModel7B):
    """DiT with CrossAttention model.

    Args:
        config (TransformerConfig): transformer config

        transformer_decoder_layer_spec (ModuleSpec): transformer layer customization specs for decoder

        pre_process (bool): Include embedding layer (used with pipeline parallelism)
        post_process (bool): Include an output layer (used with pipeline parallelism)

        fp16_lm_cross_entropy (bool, optional): Defaults to False

        parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks

        share_embeddings_and_output_weights (bool): When True, input embeddings and output logit weights are
            shared. Defaults to False.

        position_embedding_type (string): Position embedding type. Options ['learned_absolute', 'rope'].
            Defaults is 'learned_absolute'.

        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
            Defaults to 1.0 (100%). Ignored unless position_embedding_type is 'rope'.

        seq_len_interpolation_factor (float): scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
        self,
        *args,
        n_cameras=4,
        camera_condition_dim=4,
        traj_condition_dim=0,
        add_repeat_frame_embedding=True,
        concat_camera_embedding: bool = True,
        concat_traj_embedding=False,
        in_channels=16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_cameras = n_cameras
        self.camera_condition_dim = camera_condition_dim
        self.concat_camera_embedding = concat_camera_embedding
        self.traj_condition_dim = traj_condition_dim
        self.concat_traj_embedding = concat_traj_embedding
        self.add_repeat_frame_embedding = add_repeat_frame_embedding

        if self.concat_camera_embedding:
            in_channels = in_channels + camera_condition_dim if camera_condition_dim > 0 else in_channels

        if self.concat_traj_embedding:
            in_channels = in_channels + traj_condition_dim if traj_condition_dim > 0 else in_channels

        in_channels = in_channels + 1 if self.concat_padding_mask else in_channels

        self.x_embedder = (
            PatchEmbed(
                spatial_patch_size=self.patch_spatial,
                temporal_patch_size=self.patch_temporal,
                in_channels=in_channels,
                out_channels=self.config.hidden_size,
                bias=False,
                keep_spatio=True,
                legacy_patch_emb=self.legacy_patch_emb,
            )
            .cuda()
            .to(dtype=torch.bfloat16)
        )

        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder = MultiCameraSinCosPosEmbAxis(
                h_extrapolation_ratio=1,
                w_extrapolation_ratio=1,
                t_extrapolation_ratio=1,
                model_channels=self.config.hidden_size,
                len_h=self.max_img_h // self.patch_spatial,
                len_w=self.max_img_w // self.patch_spatial,
                len_t=self.max_frames // self.patch_temporal,
                interpolation=self.pos_emb_interpolation,
            )

        self.view_embeddings = nn.Embedding(n_cameras, camera_condition_dim)  # Learnable embedding layer
        if self.concat_traj_embedding:
            self.traj_embeddings = nn.Linear(192, self.traj_condition_dim)  # Learnable embedding layer
        if self.add_repeat_frame_embedding:
            self.repeat_frame_embedding = nn.Linear(1, camera_condition_dim)  # Learnable embedding layer

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        trajectory: Optional[torch.Tensor] = None,
        frame_repeat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.concat_padding_mask:
            padding_mask = padding_mask.squeeze(0)
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )

        # ********* multicamera *********
        view_indices = torch.arange(self.n_cameras).to(x_B_C_T_H_W.device)  # View indices [0, 1, ..., V-1]
        view_embedding = self.view_embeddings(view_indices)  # Shape: [V, embedding_dim]
        view_embedding = rearrange(view_embedding, "V D -> D V")
        view_embedding = (
            view_embedding.unsqueeze(0).unsqueeze(3).unsqueeze(4).unsqueeze(5)
        )  # Shape: [1, D, V, 1, 1, 1]

        if self.add_repeat_frame_embedding:
            if frame_repeat is None:
                frame_repeat = (
                    torch.zeros([x_B_C_T_H_W.shape[0], view_embedding.shape[1]])
                    .to(view_embedding.device)
                    .to(view_embedding.dtype)
                )
            frame_repeat_embedding = self.repeat_frame_embedding(frame_repeat.unsqueeze(-1))
            frame_repeat_embedding = rearrange(frame_repeat_embedding, "B V D -> B D V")
            view_embedding = view_embedding + frame_repeat_embedding.unsqueeze(3).unsqueeze(4).unsqueeze(5)

        x_B_C_V_T_H_W = rearrange(x_B_C_T_H_W, "B C (V T) H W -> B C V T H W", V=self.n_cameras)
        view_embedding = view_embedding.expand(
            x_B_C_V_T_H_W.shape[0],
            view_embedding.shape[1],
            view_embedding.shape[2],
            x_B_C_V_T_H_W.shape[3],
            x_B_C_V_T_H_W.shape[4],
            x_B_C_V_T_H_W.shape[5],
        )  # Shape: [B, V, 3, t, H, W]
        if self.concat_traj_embedding:
            traj_emb = self.traj_embeddings(trajectory)
            traj_emb = traj_emb.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)
            traj_emb = traj_emb.expand(
                x_B_C_V_T_H_W.shape[0],
                traj_emb.shape[1],
                view_embedding.shape[2],
                x_B_C_V_T_H_W.shape[3],
                x_B_C_V_T_H_W.shape[4],
                x_B_C_V_T_H_W.shape[5],
            )  # Shape: [B, V, 3, t, H, W]

            x_B_C_V_T_H_W = torch.cat([x_B_C_V_T_H_W, view_embedding, traj_emb], dim=1)
        else:
            x_B_C_V_T_H_W = torch.cat([x_B_C_V_T_H_W, view_embedding], dim=1)

        x_B_C_T_H_W = rearrange(x_B_C_V_T_H_W, " B C V T H W -> B C (V T) H W", V=self.n_cameras)
        # ********* multicamera *********

        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)
        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            if extra_pos_emb is not None:
                extra_pos_emb = rearrange(extra_pos_emb, "B T H W D -> (T H W) B D")
                return x_B_T_H_W_D, [self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb]
            else:
                return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps)

        if "fps_aware" in self.pos_emb_cls:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D, fps=fps.cuda())  # [B, T, H, W, D]
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]
        return x_B_T_H_W_D, None

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        crossattn_emb: Tensor,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        pos_ids: Tensor = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): vae encoded videos (b s c)
            encoder_decoder_attn_mask (Tensor): cross-attention mask between encoder and decoder
            inference_params (InferenceParams): relevant arguments for inferencing

        Returns:
            Tensor: loss tensor
        """
        # Decoder forward
        # Decoder embedding.
        # print(f'x={x}')
        # x = x.squeeze(0)
        original_shape = x.shape
        B, C, T, H, W = original_shape

        fps = kwargs.get('fps', None)
        if len(fps.shape) > 1:
            fps = fps.squeeze(0)
        padding_mask = kwargs.get('padding_mask', None)
        image_size = kwargs.get('image_size', None)
        trajectory = kwargs.get('trajectory', None)
        frame_repeat = kwargs.get('frame_repeat', None)

        if self.pre_process:
            x_B_T_H_W_D, rope_emb_L_1_1_D = self.prepare_embedded_sequence(
                x, fps=fps, padding_mask=padding_mask, trajectory=trajectory, frame_repeat=frame_repeat
            )
            B, T, H, W, D = x_B_T_H_W_D.shape
            # print(f'x_T_H_W_B_D.shape={x_T_H_W_B_D.shape}')
            x_S_B_D = rearrange(x_B_T_H_W_D, "B T H W D -> (T H W) B D")
            # print(f'x_S_B_D.shape={x_S_B_D.shape}')
        else:
            # intermediate stage of pipeline
            x_S_B_D = None  # should it take encoder_hidden_states

        _, _, D = x_S_B_D.shape

        # print(f'x_S_B_D={x_S_B_D}')

        # logging affline scale information
        affline_scale_log_info = {}

        timesteps_B_D, adaln_lora_B_3D = self.t_embedder(timesteps.flatten())
        affline_emb_B_D = timesteps_B_D
        affline_scale_log_info["timesteps_B_D"] = timesteps_B_D.detach()

        if self.additional_timestamp_channels:
            if type(image_size) == tuple:
                image_size = image_size[0]
            additional_cond_B_D = self.prepare_additional_timestamp_embedder(
                bs=x.shape[0],
                fps=fps,
                h=image_size[:, 0],
                w=image_size[:, 1],
                org_h=image_size[:, 2],
                org_w=image_size[:, 3],
            )

            affline_emb_B_D += additional_cond_B_D
            affline_scale_log_info["additional_cond_B_D"] = additional_cond_B_D.detach()

        affline_scale_log_info["affline_emb_B_D"] = affline_emb_B_D.detach()
        affline_emb_B_D = self.affline_norm(affline_emb_B_D)

        crossattn_emb = rearrange(crossattn_emb, 'B S D -> S B D')

        # [Parth] Enable Sequence Parallelism
        if self.config.sequence_parallel:
            if self.pre_process:
                x_S_B_D = tensor_parallel.scatter_to_sequence_parallel_region(x_S_B_D)
                if len(rope_emb_L_1_1_D) > 1:
                    rope_emb_L_1_1_D[1] = tensor_parallel.scatter_to_sequence_parallel_region(rope_emb_L_1_1_D[1])
            crossattn_emb = tensor_parallel.scatter_to_sequence_parallel_region(crossattn_emb)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if self.config.clone_scatter_output_in_embedding:
                if self.pre_process:
                    x_S_B_D = x_S_B_D.clone()
                    rope_emb_L_1_1_D[1] = rope_emb_L_1_1_D[1].clone()
                crossattn_emb = crossattn_emb.clone()

        packed_seq_params = {
            'adaln_lora_B_3D': adaln_lora_B_3D.detach(),
            'extra_pos_emb': rope_emb_L_1_1_D[1].detach(),
        }
        x_S_B_D = self.decoder(
            hidden_states=x_S_B_D,
            attention_mask=affline_emb_B_D,
            context=crossattn_emb,
            context_mask=None,
            packed_seq_params=packed_seq_params,
            rotary_pos_emb=rope_emb_L_1_1_D[0],
        )
        # Return if not post_process
        if not self.post_process:
            return x_S_B_D

        if self.config.sequence_parallel:
            x_S_B_D = tensor_parallel.gather_from_sequence_parallel_region(x_S_B_D)

        x_B_T_H_W_D = rearrange(x_S_B_D, "(T H W) B D -> B T H W D", B=B, T=T, H=H, W=W, D=D)
        x_B_T_H_W_D = self.decoder_head(x_B_T_H_W_D, affline_emb_B_D, None, original_shape, None, adaln_lora_B_3D)

        return x_B_T_H_W_D

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility (removing extra state).

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        sharded_state_dict[f"{prefix}x_embedder.proj.1.weight"].allow_shape_mismatch = True

        return sharded_state_dict


class VideoExtendMultiCameraDiTCrossAttentionModel7B(MultiCameraDiTCrossAttentionModel7B):
    def __init__(self, *args, in_channels=16, **kwargs):
        # extra channel for video condition mask
        super().__init__(*args, in_channels=in_channels + 1, **kwargs)
        logging.info(f"VideoExtendGeneralDIT in_channels: {in_channels + 1}")

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        data_type: Optional[DataType] = DataType.VIDEO,
        video_cond_bool: Optional[torch.Tensor] = None,
        condition_video_indicator: Optional[torch.Tensor] = None,
        condition_video_input_mask: Optional[torch.Tensor] = None,
        condition_video_augment_sigma: Optional[torch.Tensor] = None,
        condition_video_pose: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Args:
        condition_video_augment_sigma: (B) tensor of sigma value for the conditional input augmentation
        condition_video_pose: (B, 1, T, H, W) tensor of pose condition
        """
        B, C, T, H, W = x.shape

        if data_type == DataType.VIDEO:
            assert (
                condition_video_input_mask is not None
            ), "condition_video_input_mask is required for video data type; check if your model_obj is extend_model.FSDPDiffusionModel or the base DiffusionModel"
            if self.cp_group is not None:
                condition_video_input_mask = rearrange(
                    condition_video_input_mask, "B C (V T) H W -> B C V T H W", V=self.n_cameras
                )
                condition_video_input_mask = split_inputs_cp(
                    condition_video_input_mask, seq_dim=3, cp_group=self.cp_group
                )
                condition_video_input_mask = rearrange(
                    condition_video_input_mask, "B C V T H W -> B C (V T) H W", V=self.n_cameras
                )
            input_list = [x, condition_video_input_mask]
            if condition_video_pose is not None:
                if condition_video_pose.shape[2] > T:
                    logging.warning(
                        f"condition_video_pose has more frames than the input video: {condition_video_pose.shape} > {x.shape}"
                    )
                    condition_video_pose = condition_video_pose[:, :, :T, :, :].contiguous()
                input_list.append(condition_video_pose)
            x = torch.cat(
                input_list,
                dim=1,
            )

        return super().forward(
            x=x,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            condition_video_augment_sigma=condition_video_augment_sigma,
            **kwargs,
        )


def dit_data_step_no_split_on_cp(module, dataloader_iter):
    batch = next(dataloader_iter)[0]

    batch = {k: v.to(device='cuda', non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

    batch["is_preprocessed"] = True  # assume data is preprocessed
    if ('seq_len_q' in batch) and ('seq_len_kv' in batch):
        cu_seqlens = batch['seq_len_q'].cumsum(dim=0).to(torch.int32)
        zero = torch.zeros(1, dtype=torch.int32, device="cuda")
        cu_seqlens = torch.cat((zero, cu_seqlens))

        cu_seqlens_kv = batch['seq_len_kv'].cumsum(dim=0).to(torch.int32)
        cu_seqlens_kv = torch.cat((zero, cu_seqlens_kv))

        batch['packed_seq_params'] = {
            'self_attention': PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                qkv_format=module.qkv_format,
            ),
            'cross_attention': PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens_kv,
                qkv_format=module.qkv_format,
            ),
        }

    return batch


@dataclass
class MultiCameraDiT7BConfig(DiT7BConfig):
    n_cameras: int = 4
    camera_condition_dim: int = 4
    traj_condition_dim: int = 0
    add_repeat_frame_embedding: bool = True
    concat_camera_embedding: bool = True
    concat_traj_embedding: bool = False
    pixel_chunk_duration: int = 57
    data_step_fn = dit_data_step_no_split_on_cp

    @override
    def configure_model(self, tokenizer=None) -> MultiCameraDiTCrossAttentionModel7B:
        return MultiCameraDiTCrossAttentionModel7B(
            self,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            n_cameras=self.n_cameras,
            camera_condition_dim=self.camera_condition_dim,
            traj_condition_dim=self.traj_condition_dim,
            add_repeat_frame_embedding=self.add_repeat_frame_embedding,
            concat_camera_embedding=self.concat_camera_embedding,
            concat_traj_embedding=self.concat_traj_embedding,
        )

    def configure_vae(self):
        return dynamic_import(self.vae_module)(self.vae_path, pixel_chunk_duration=self.pixel_chunk_duration)


@dataclass
class VideoExtendMultiCameraDiT7BConfig(MultiCameraDiT7BConfig):
    model_name = ('cosmos_7b_video2world',)

    @override
    def configure_model(self, tokenizer=None) -> VideoExtendMultiCameraDiTCrossAttentionModel7B:
        return VideoExtendMultiCameraDiTCrossAttentionModel7B(
            self,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            n_cameras=self.n_cameras,
            camera_condition_dim=self.camera_condition_dim,
            traj_condition_dim=self.traj_condition_dim,
            add_repeat_frame_embedding=self.add_repeat_frame_embedding,
            concat_camera_embedding=self.concat_camera_embedding,
            concat_traj_embedding=self.concat_traj_embedding,
        )


class FrameRepeatAttr(TrainingOnlyEmbModel):
    def __init__(self):
        super().__init__()

    def forward(self, frame_repeat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "frame_repeat": frame_repeat / 10.0,
        }

    def details(self) -> str:
        return "Frame repeat, Output key: [frame_repeat]"


@attrs.define(slots=False)
class FrameRepeatConfig:
    obj: Any = FrameRepeatAttr()  # No arguments
    dropout_rate: float = 0.0
    input_key: str = "frame_repeat"


class TrajectoryAttr(AbstractEmbModel):
    def __init__(self, traj_dim: int):
        super().__init__()
        self.traj_dim = traj_dim

    def forward(self, traj: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "trajectory": traj,
        }

    def details(self) -> str:
        return f"Traj dim : {self.traj_dim} \n\tOutput key: [trajectory]"


@attrs.define(slots=False)
class TrajectoryConfig:
    # context_dim == t5_text_embeddings_dim
    obj: Any = TrajectoryAttr(traj_dim=192)
    dropout_rate: float = 0.2
    input_key: str = "trajectory"


class MultiCamCosmosDiffusionPipeline(CosmosDiffusionPipeline):
    def __init__(self, n_cameras, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_cameras = n_cameras

    # @torch.no_grad()
    # def encode(self, state: torch.Tensor) -> torch.Tensor:
    #     state = rearrange(state, "B C (V T) H W -> (B V) C T H W", V=self.n_cameras)
    #     encoded_state = self.vae.encode(state)
    #     encoded_state = rearrange(encoded_state, "(B V) C T H W -> B C (V T) H W", V=self.n_cameras) * self.sigma_data
    #     return encoded_state

    # @torch.no_grad()
    # def decode(self, latent: torch.Tensor) -> torch.Tensor:
    #     latent = rearrange(latent, "B C (V T) H W -> (B V) C T H W", V=self.n_cameras)
    #     decoded_state = self.vae.decode(latent / self.sigma_data)
    #     decoded_state = rearrange(decoded_state, "(B V) C T H W -> B C (V T) H W", V=self.n_cameras)
    #     return decoded_state

    def compute_loss_with_epsilon_and_sigma(
        self,
        data_batch: dict[str, torch.Tensor],
        x0_from_data_batch: torch.Tensor,
        x0: torch.Tensor,
        condition: Edify4Condition,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
    ):
        # Only support video batch
        if parallel_state.is_initialized():
            if parallel_state.get_context_parallel_world_size() > 1:
                # Turn on CP
                cp_group = parallel_state.get_context_parallel_group()

                x0 = rearrange(x0, "B C (V T) H W -> (B V) C T H W", V=self.n_cameras)
                epsilon = rearrange(epsilon, "B C (V T) H W -> (B V) C T H W", V=self.n_cameras)
                x0 = split_inputs_cp(x=x0, seq_dim=2, cp_group=cp_group)
                epsilon = split_inputs_cp(x=epsilon, seq_dim=2, cp_group=cp_group)

                x0 = rearrange(x0, "(B V) C T H W -> B C (V T) H W", V=self.n_cameras)
                epsilon = rearrange(epsilon, "(B V) C T H W -> B C (V T) H W", V=self.n_cameras)

        output_batch, kendall_loss, pred_mse, edm_loss = super().compute_loss_with_epsilon_and_sigma(
            data_batch, x0_from_data_batch, x0, condition, epsilon, sigma
        )
        if not self.is_image_batch(data_batch):
            if self.loss_reduce == "sum" and parallel_state.get_context_parallel_world_size() > 1:
                kendall_loss *= parallel_state.get_context_parallel_world_size()

        return output_batch, kendall_loss, pred_mse, edm_loss

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        solver_option: COMMON_SOLVER_OPTIONS = "2ab",
    ) -> Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        """

        is_image_batch = self.is_image_batch(data_batch)
        if n_sample is None:
            input_key = self.input_image_key if is_image_batch else self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            if is_image_batch:
                state_shape = (self.state_shape[0], 1, *self.state_shape[2:])  # C,T,H,W

        cp_enabled = parallel_state.get_context_parallel_world_size() > 1

        if self._noise_generator is None:
            self._initialize_generators()

        x0_fn = self.get_x0_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)

        state_shape = list(state_shape)

        if len(state_shape) == 4:
            state_shape = [1] + state_shape
        np.random.seed(self.seed)
        x_sigma_max = (
            torch.from_numpy(np.random.randn(*state_shape).astype(np.float32)).to(
                dtype=torch.float32, device=self.tensor_kwargs["device"]
            )
            * self.sde.sigma_max
        )

        if cp_enabled:
            cp_group = parallel_state.get_context_parallel_group()
            x_sigma_max = rearrange(x_sigma_max, "B C (V T) H W -> (B V) C T H W", V=self.n_cameras)
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=cp_group)
            x_sigma_max = rearrange(x_sigma_max, "(B V) C T H W -> B C (V T) H W", V=self.n_cameras)

        if self.sampler_type == "EDM":
            samples = self.sampler(x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=self.sde.sigma_max)
        elif self.sampler_type == "RES":
            samples = self.sampler(
                x0_fn, x_sigma_max, sigma_max=self.sde.sigma_max, num_steps=num_steps, solver_option=solver_option
            )

        if cp_enabled:
            cp_group = parallel_state.get_context_parallel_group()
            samples = rearrange(samples, "B C (V T) H W -> (B V) C T H W", V=self.n_cameras)
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=cp_group)
            samples = rearrange(samples, "(B V) C T H W -> B C (V T) H W", V=self.n_cameras)

        return samples

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        # @TODO(tylerz): Why don't we need to broadcast the condition and uncondition tensors here?
        # to_cp = parallel_state.get_context_parallel_world_size() > 1

        # # For inference, check if parallel_state is initialized
        # if parallel_state.is_initialized():
        #     condition = broadcast_condition(condition, to_tp=True, to_cp=to_cp)
        #     uncondition = broadcast_condition(uncondition, to_tp=True, to_cp=to_cp)
        # else:
        #     assert not to_cp, "parallel_state is not initialized, context parallel should be turned off."

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0, _, _ = self.denoise(noise_x, sigma, condition)
            uncond_x0, _, _ = self.denoise(noise_x, sigma, uncondition)
            return cond_x0 + guidance * (cond_x0 - uncond_x0)

        return x0_fn


class MultiCameraDiTModel(DiTModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vae = None
        self.conditioner = VideoConditioner(
            text=TextConfig(),
            fps=FPSConfig(),
            num_frames=NumFramesConfig(),
            image_size=ImageSizeConfig(),
            padding_mask=PaddingMaskConfig(),
            frame_repeat=FrameRepeatConfig(),
            trajectory=TrajectoryConfig(),
        )
        self.diffusion_pipeline = MultiCamCosmosDiffusionPipeline(
            net=self,
            n_cameras=self.config.n_cameras,
            conditioner=self.conditioner,
            loss_add_logvar=self.config.loss_add_logvar,
        )

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        state = rearrange(state, "B C (V T) H W -> (B V) C T H W", V=self.config.n_cameras)
        encoded_state = self.vae.encode(state)
        encoded_state = rearrange(encoded_state, "(B V) C T H W -> B C (V T) H W", V=self.config.n_cameras)
        return encoded_state

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        latent = rearrange(latent, "B C (V T) H W -> (B V) C T H W", V=self.config.n_cameras)
        decoded_state = self.vae.decode(latent / self.config.sigma_data)
        decoded_state = rearrange(decoded_state, "(B V) C T H W -> B C (V T) H W", V=self.config.n_cameras)
        return decoded_state

    @torch.no_grad()
    def data_step(self, dataloader_iter):
        batch = super().data_step(dataloader_iter)
        batch = {k: v.to(dtype=torch.bfloat16) if torch.is_tensor(v) else v for k, v in batch.items()}

        if self.vae is None:
            self.vae = self.config.configure_vae()
            self.vae.to('cuda')

        batch['video'] = self.encode(batch['video'])
        seq_len = batch['video'].shape[-1] * batch['video'].shape[-2] * batch['video'].shape[-3]
        batch["loss_mask"] = torch.ones(seq_len, dtype=torch.bfloat16, device=batch['video'].device)
        from megatron.core import mpu

        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()

        if cp_size > 1:
            num_valid_tokens_in_ub = None
            if 'loss_mask' in batch and batch['loss_mask'] is not None:
                num_valid_tokens_in_ub = batch['loss_mask'].sum()
            batch['num_valid_tokens_in_ub'] = num_valid_tokens_in_ub
            batch["loss_mask"] = (
                batch["loss_mask"].view(cp_size, batch["loss_mask"].shape[0] // cp_size)[cp_rank, ...].contiguous()
            )

        return batch

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        state_shape = batch['video'].shape
        sample = self.diffusion_pipeline.generate_samples_from_batch(
            batch,
            guidance=7,
            state_shape=state_shape,
            num_steps=35,
            is_negative_prompt=True if 'neg_t5_text_embeddings' in batch else False,
        )

        video = (1.0 + self.decode(sample)).clamp(0, 2) / 2  # [B, 3, T, H, W]

        video = (video * 255).to(torch.uint8).cpu().numpy().astype(np.uint8)

        video_segments = einops.rearrange(video, "b c (v t) h w -> b c v t h w", v=self.config.n_cameras)
        if self.config.n_cameras == 4:
            grid_video = einops.rearrange(video_segments, "b c (h w) t h1 w1 -> b c t (h h1) (w w1)", h=2, w=2)
        elif self.config.n_cameras == 6:
            grid_video = torch.stack(
                [
                    torch.from_numpy(video_segments[:, :, 1]),
                    torch.from_numpy(video_segments[:, :, 0]),
                    torch.from_numpy(video_segments[:, :, 2]),
                    torch.from_numpy(video_segments[:, :, 4]),
                    torch.from_numpy(video_segments[:, :, 3]),
                    torch.from_numpy(video_segments[:, :, 5]),
                ],
                dim=2,
            )
            grid_video = einops.rearrange(grid_video, "b c (h w) t h1 w1 -> b c t (h h1) (w w1)", h=2, w=3)

        grid_video = rearrange(grid_video, "b c t h w -> b t c h w")

        # wandb is on the last rank for megatron, first rank for nemo
        wandb_rank = 0

        if parallel_state.get_data_parallel_src_rank() == wandb_rank:
            if torch.distributed.get_rank() == wandb_rank:
                gather_list = [None for _ in range(parallel_state.get_data_parallel_world_size())]
            else:
                gather_list = None
            torch.distributed.gather_object(
                grid_video, gather_list, wandb_rank, group=parallel_state.get_data_parallel_group()
            )
            if gather_list is not None:
                videos = []
                for grid_video in gather_list:
                    grid_video = grid_video.numpy()
                    videos.append(wandb.Video(grid_video, fps=12))
                wandb.log({'prediction': videos}, step=self.global_step)

        return None

    def on_validation_end(self):
        pass
