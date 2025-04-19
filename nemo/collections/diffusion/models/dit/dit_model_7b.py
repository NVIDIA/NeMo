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

# pylint: disable=C0115,C0116,C0301

import math
import warnings
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_sharded_tensor_for_checkpoint
from torch import Tensor
from torch.autograd import Function
from torch.distributed import ProcessGroup, all_gather, get_process_group_ranks, get_world_size
from torchvision import transforms

from nemo.collections.diffusion.models.dit.action_control.action_control_layers import ActionControlTorchMlp
from nemo.collections.diffusion.models.dit.cosmos_layer_spec import (
    get_dit_adaln_block_with_transformer_engine_spec as DiTLayerWithAdaLNspec,
)
from nemo.collections.diffusion.sampler.conditioner import DataType


def gather_along_first_dim(tensor, process_group):
    return AllGather.apply(tensor, process_group)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SDXLTimesteps(nn.Module):
    def __init__(self, num_channels: int = 320):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        in_dype = timesteps.dtype
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return emb.to(in_dype)


class SDXLTimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora_B_3D = emb
            emb_B_D = sample
        else:
            emb_B_D = emb
            adaln_lora_B_3D = None

        return emb_B_D, adaln_lora_B_3D


class AllGather(Function):
    @staticmethod
    def forward(ctx, tensor, process_group):
        world_size = dist.get_world_size(process_group)
        ctx.world_size = world_size
        ctx.rank = process_group.rank()

        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor.contiguous(), process_group)
        return torch.cat(gathered_tensors, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        world_size = ctx.world_size
        rank = ctx.rank

        # Split the gradient tensor
        grad_chunks = grad_output.chunk(world_size)

        # Select the gradient chunk for the current rank
        grad_input = grad_chunks[rank]
        return grad_input, None


class PatchEmbed(nn.Module):
    # TODO: (qsh 2024-09-20) update docstring
    """
    PatchEmbed is a module for embedding patches from an input tensor by applying either 3D or 2D convolutional layers,
    depending on the . This module can process inputs with temporal (video) and spatial (image) dimensions,
    making it suitable for video and image processing tasks. It supports dividing the input into patches and embedding each
    patch into a vector of size `out_channels`.

    Parameters:
    - spatial_patch_size (int): The size of each spatial patch.
    - temporal_patch_size (int): The size of each temporal patch.
    - in_channels (int): Number of input channels. Default: 3.
    - out_channels (int): The dimension of the embedding vector for each patch. Default: 768.
    - bias (bool): If True, adds a learnable bias to the output of the convolutional layers. Default: True.
    - keep_spatio (bool): If True, the spatial dimensions are kept separate in the output tensor, otherwise, they are flattened. Default: False.
    - legacy_patch_emb (bool): If True, applies 3D convolutional layers for video inputs, otherwise, use Linear! The legacy model is for backward compatibility. Default: True.
    The output shape of the module depends on the `keep_spatio` flag. If `keep_spatio`=True, the output retains the spatial dimensions.
    Otherwise, the spatial dimensions are flattened into a single dimension.
    """

    def __init__(
        self,
        spatial_patch_size,
        temporal_patch_size,
        in_channels=3,
        out_channels=768,
        bias=True,
        keep_spatio=False,
        legacy_patch_emb: bool = True,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        assert keep_spatio, "Only support keep_spatio=True"
        self.keep_spatio = keep_spatio
        self.legacy_patch_emb = legacy_patch_emb

        if legacy_patch_emb:
            self.proj = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
                stride=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
                bias=bias,
            )
            self.out = Rearrange("b c t h w -> b t h w c")
        else:
            self.proj = nn.Sequential(
                Rearrange(
                    "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                    r=temporal_patch_size,
                    m=spatial_patch_size,
                    n=spatial_patch_size,
                ),
                nn.Linear(
                    in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size,
                    out_channels,
                    bias=bias,
                ),
            )
            self.out = nn.Identity()

    def forward(self, x):
        """
        Forward pass of the PatchEmbed module.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (B, C, T, H, W) where
            B is the batch size,
            C is the number of channels,
            T is the temporal dimension,
            H is the height, and
            W is the width of the input.

        Returns:
        - torch.Tensor: The embedded patches as a tensor, with shape b t h w c.
        """
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return self.out(x)


class FourierFeatures(nn.Module):
    """
    Implements a layer that generates Fourier features from input tensors, based on randomly sampled
    frequencies and phases. This can help in learning high-frequency functions in low-dimensional problems.

    [B] -> [B, D]

    Parameters:
        num_channels (int): The number of Fourier features to generate.
        bandwidth (float, optional): The scaling factor for the frequency of the Fourier features. Defaults to 1.
        normalize (bool, optional): If set to True, the outputs are scaled by sqrt(2), usually to normalize
                                    the variance of the features. Defaults to False.

    Example:
        >>> layer = FourierFeatures(num_channels=256, bandwidth=0.5, normalize=True)
        >>> x = torch.randn(10, 256)  # Example input tensor
        >>> output = layer(x)
        >>> print(output.shape)  # Expected shape: (10, 256)
    """

    def __init__(self, num_channels, bandwidth=1, normalize=False):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * bandwidth * torch.randn(num_channels), persistent=True)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels), persistent=True)
        self.gain = np.sqrt(2) if normalize else 1

    def forward(self, x, gain: float = 1.0):
        """
        Apply the Fourier feature transformation to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            gain (float, optional): An additional gain factor applied during the forward pass. Defaults to 1.

        Returns:
            torch.Tensor: The transformed tensor, with Fourier features applied.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32).ger(self.freqs.to(torch.float32)).add(self.phases.to(torch.float32))
        x = x.cos().mul(self.gain * gain).to(in_dtype)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of video DiT.
    """

    def __init__(
        self,
        hidden_size,
        spatial_patch_size,
        temporal_patch_size,
        out_channels,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False)
            )

        self.sequence_parallel = getattr(parallel_state, "sequence_parallel", False)

    def forward(
        self,
        x_BT_HW_D,
        emb_B_D,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ):
        if self.use_adaln_lora:
            assert adaln_lora_B_3D is not None
            shift_B_D, scale_B_D = (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D[:, : 2 * self.hidden_size]).chunk(
                2, dim=1
            )
        else:
            shift_B_D, scale_B_D = self.adaLN_modulation(emb_B_D).chunk(2, dim=1)

        B = emb_B_D.shape[0]
        T = x_BT_HW_D.shape[0] // B
        shift_BT_D, scale_BT_D = repeat(shift_B_D, "b d -> (b t) d", t=T), repeat(scale_B_D, "b d -> (b t) d", t=T)
        x_BT_HW_D = modulate(self.norm_final(x_BT_HW_D), shift_BT_D, scale_BT_D)
        if self.sequence_parallel:
            x_T_B_HW_D = rearrange(x_BT_HW_D, "(b t) hw d -> t b hw d", b=B, t=T)
            x_T_B_HW_D = gather_along_first_dim(x_T_B_HW_D, parallel_state.get_tensor_model_parallel_group())
            x_BT_HW_D = rearrange(x_T_B_HW_D, "t b hw d -> (b t) hw d", b=B)

        x_BT_HW_D = self.linear(x_BT_HW_D)
        return x_BT_HW_D


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def split_inputs_cp(x: Tensor, seq_dim: int, cp_group: ProcessGroup) -> Tensor:
    """
    Split input tensor along the sequence dimension for checkpoint parallelism.

    This function divides the input tensor into equal parts along the specified
    sequence dimension, based on the number of ranks in the checkpoint parallelism group.
    It then selects the part corresponding to the current rank.

    Args:
        x: Input tensor to be split.
        seq_dim: The dimension along which to split the input (sequence dimension).
        cp_group: The process group for checkpoint parallelism.

    Returns:
        A slice of the input tensor corresponding to the current rank.

    Raises:
        AssertionError: If the sequence dimension is not divisible by the number of ranks.
    """
    cp_ranks = get_process_group_ranks(cp_group)
    cp_size = len(cp_ranks)

    assert x.shape[seq_dim] % cp_size == 0, f"{x.shape[seq_dim]} cannot divide cp_size {cp_size}"
    x = x.view(*x.shape[:seq_dim], cp_size, x.shape[seq_dim] // cp_size, *x.shape[(seq_dim + 1) :])
    seq_idx = torch.tensor([cp_group.rank()], device=x.device)
    x = x.index_select(seq_dim, seq_idx)
    # Note that the new sequence length is the original sequence length / cp_size
    x = x.view(*x.shape[:seq_dim], -1, *x.shape[(seq_dim + 2) :])
    return x


def cat_outputs_cp(x: Tensor, seq_dim: int, cp_group: ProcessGroup) -> Tensor:
    """
    Concatenates tensors from multiple processes along a specified dimension.

    This function gathers tensors from all processes in the given process group
    and concatenates them along the specified dimension.

    Args:
        x (Tensor): The input tensor to be gathered and concatenated.
        seq_dim (int): The dimension along which to concatenate the gathered tensors.
        cp_group (ProcessGroup): The process group containing all the processes involved in the gathering.

    Returns:
        Tensor: A tensor resulting from the concatenation of tensors from all processes.

    Raises:
        RuntimeError: If the gathering of tensors fails.
    """
    # Number of processes in the group
    world_size = get_world_size(cp_group)

    # List to hold tensors from each rank
    gathered_tensors = [torch.zeros_like(x) for _ in range(world_size)]

    # Attempt to gather tensors from all ranks
    try:
        all_gather(gathered_tensors, x, group=cp_group)
    except RuntimeError as e:
        raise RuntimeError(f"Gathering failed: {e}")

    # Concatenate tensors along the specified dimension
    return torch.cat(gathered_tensors, dim=seq_dim)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def normalize(x: torch.Tensor, dim: Optional[List[int]] = None, eps: float = 0) -> torch.Tensor:
    """
    Normalizes the input tensor along specified dimensions such that the average square norm of elements is adjusted.

    Args:
        x (torch.Tensor): The input tensor to normalize.
        dim (list, optional): The dimensions over which to normalize. If None, normalizes over all dimensions except the first.
        eps (float, optional): A small constant to ensure numerical stability during division.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class VideoPositionEmb(nn.Module):
    def __init__(self):
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
            if isinstance(self, VideoRopePosition3DEmb):
                seq_dim = 0
            else:
                seq_dim = 1
            embeddings = split_inputs_cp(x=embeddings, seq_dim=seq_dim, cp_group=self.cp_group)
        return embeddings

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor]):
        raise NotImplementedError


class VideoRopePosition3DEmb(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrp_ratio: float = 1.0,
        w_extrp_ratio: float = 1.0,
        t_extrp_ratio: float = 1.0,
        **kwargs,  # used for compatibility with other positional embeddings; unused in this class
    ):
        del kwargs
        super().__init__()
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float))
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w

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

        self.h_ntk_factor = h_extrp_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrp_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrp_ratio ** (dim_t / (dim_t - 2))

    def generate_embeddings(
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
        assert (
            B == 1 or T == 1
        ), "Batch size should be 1 or T should be 1. Image batch should have T=1, while video batch should have B=1."
        assert (
            H <= self.max_h and W <= self.max_w
        ), f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w}) configured for positional embedding. Please adjust the input size or increase the maximum dimensions in the model configuration."
        self.seq = self.seq.cuda()
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

        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()


class SinCosPosEmbAxis(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        interpolation: str,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        is_learnable: bool = True,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        **kwargs,
    ):
        # TODO: (qsh 2024-11-08) add more interpolation methods and args for extrapolation fine-tuning
        """
        Args:
            interpolation (str): we curretly only support "crop", ideally when we need extrapolation capacity, we should adjust frequency or other more advanced methods. they are not implemented yet.
        """
        del kwargs  # unused
        super().__init__()
        self.interpolation = interpolation
        self.is_learnable = is_learnable
        assert self.interpolation in ["crop"], f"Unknown interpolation method {self.interpolation}"

        self.pos_emb_h = nn.Parameter(torch.zeros(len_h, model_channels))
        self.pos_emb_w = nn.Parameter(torch.zeros(len_w, model_channels))
        self.pos_emb_t = nn.Parameter(torch.zeros(len_t, model_channels))

        trunc_normal_(self.pos_emb_h, std=0.02)
        trunc_normal_(self.pos_emb_w, std=0.02)
        trunc_normal_(self.pos_emb_t, std=0.02)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor]) -> torch.Tensor:
        B, T, H, W, C = B_T_H_W_C
        if self.interpolation == "crop":
            emb_h_H = self.pos_emb_h[:H]
            emb_w_W = self.pos_emb_w[:W]
            emb_t_T = self.pos_emb_t[:T]
            emb = (
                repeat(emb_t_T, "t d-> b t h w d", b=B, h=H, w=W)
                + repeat(emb_h_H, "h d-> b t h w d", b=B, t=T, w=W)
                + repeat(emb_w_W, "w d-> b t h w d", b=B, t=T, h=H)
            )
            assert list(emb.shape)[:4] == [B, T, H, W], f"bad shape: {list(emb.shape)[:4]} != {B, T, H, W}"
        else:
            raise ValueError(f"Unknown interpolation method {self.interpolation}")

        return normalize(emb, dim=-1, eps=1e-6)


class DiTCrossAttentionModel7B(VisionModule):
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
        config: TransformerConfig,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        position_embedding_type: Literal["learned_absolute", "rope"] = "rope",
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[float] = None,
    ):

        super(DiTCrossAttentionModel7B, self).__init__(config=config)

        self.config: TransformerConfig = config
        from megatron.core.enums import ModelType

        self.model_type = ModelType.encoder_or_decoder

        self.transformer_decoder_layer_spec = DiTLayerWithAdaLNspec()
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = True
        self.add_decoder = True
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.position_embedding_type = position_embedding_type
        self.share_embeddings_and_output_weights = False
        self.additional_timestamp_channels = False
        self.concat_padding_mask = True
        self.pos_emb_cls = 'rope3d'
        self.use_adaln_lora = True
        self.patch_spatial = 2
        self.patch_temporal = 1
        self.out_channels = 16
        self.adaln_lora_dim = 256

        # Transformer decoder
        self.decoder = TransformerBlock(
            config=self.config,
            spec=self.transformer_decoder_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=False,
        )

        self.t_embedder = nn.Sequential(
            SDXLTimesteps(self.config.hidden_size),
            SDXLTimestepEmbedding(
                self.config.hidden_size, self.config.hidden_size, use_adaln_lora=self.use_adaln_lora
            ),
        )

        if self.pre_process:
            self.in_channels = 16
            self.in_channels = self.in_channels + 1 if self.concat_padding_mask else self.in_channels
            self.legacy_patch_emb = False
            self.x_embedder = (
                PatchEmbed(
                    spatial_patch_size=self.patch_spatial,
                    temporal_patch_size=self.patch_temporal,
                    in_channels=self.in_channels,
                    out_channels=self.config.hidden_size,
                    bias=False,
                    keep_spatio=True,
                    legacy_patch_emb=self.legacy_patch_emb,
                )
                .cuda()
                .to(dtype=torch.bfloat16)
            )

            self.max_img_h = 240
            self.max_img_w = 240
            self.max_frames = 128
            self.min_fps = 1
            self.max_fps = 30
            self.pos_emb_learnable = False
            self.pos_emb_interpolation = 'crop'
            self.rope_h_extrp_ratio = 1.0
            self.rope_w_extrp_ratio = 1.0
            self.rope_t_extrp_ratio = 2.0
            self.extra_per_block_abs_pos_emb = True

            self.pos_embedder = VideoRopePosition3DEmb(
                model_channels=self.config.hidden_size,
                len_h=self.max_img_h // self.patch_spatial,
                len_w=self.max_img_w // self.patch_spatial,
                len_t=self.max_frames // self.patch_temporal,
                max_fps=self.max_fps,
                min_fps=self.min_fps,
                is_learnable=self.pos_emb_learnable,
                interpolation=self.pos_emb_interpolation,
                head_dim=self.config.hidden_size // self.config.num_attention_heads,
                h_extrp_ratio=self.rope_h_extrp_ratio,
                w_extrp_ratio=self.rope_w_extrp_ratio,
                t_extrp_ratio=self.rope_t_extrp_ratio,
            )

            if self.extra_per_block_abs_pos_emb:

                self.extra_pos_embedder = SinCosPosEmbAxis(
                    h_extrapolation_ratio=1,
                    w_extrapolation_ratio=1,
                    t_extrapolation_ratio=1,
                    model_channels=self.config.hidden_size,
                    len_h=self.max_img_h // self.patch_spatial,
                    len_w=self.max_img_w // self.patch_spatial,
                    len_t=self.max_frames // self.patch_temporal,
                    interpolation=self.pos_emb_interpolation,
                )

            if parallel_state.get_context_parallel_world_size() > 1:
                cp_group = parallel_state.get_context_parallel_group()
                self.pos_embedder.enable_context_parallel(cp_group)
                self.extra_pos_embedder.enable_context_parallel(cp_group)

        if self.post_process:
            self.final_layer = FinalLayer(
                hidden_size=self.config.hidden_size,
                spatial_patch_size=self.patch_spatial,
                temporal_patch_size=self.patch_temporal,
                out_channels=self.out_channels,
                use_adaln_lora=self.use_adaln_lora,
                adaln_lora_dim=self.adaln_lora_dim,
            )

        self.build_additional_timestamp_embedder()
        # self.affline_norm = RMSNorm(self.config.hidden_size)
        import transformer_engine as te

        self.affline_norm = te.pytorch.RMSNorm(self.config.hidden_size, eps=1e-6)
        self.logvar = nn.Sequential(
            FourierFeatures(num_channels=128, normalize=True), torch.nn.Linear(128, 1, bias=False)
        )

    def build_additional_timestamp_embedder(self):
        if self.additional_timestamp_channels:
            self.additional_timestamp_channels = dict(fps=256, h=256, w=256, org_h=256, org_w=256)
            self.additional_timestamp_embedder = nn.ModuleDict()
            for cond_name, cond_emb_channels in self.additional_timestamp_channels.items():
                print(f"Building additional timestamp embedder for {cond_name} with {cond_emb_channels} channels")
                self.additional_timestamp_embedder[cond_name] = nn.Sequential(
                    SDXLTimesteps(cond_emb_channels),
                    SDXLTimestepEmbedding(cond_emb_channels, cond_emb_channels),
                )

    def prepare_additional_timestamp_embedder(self, **kwargs):
        condition_concat = []
        for cond_name, embedder in self.additional_timestamp_embedder.items():
            condition_concat.append(embedder(kwargs[cond_name]))
        embedding = torch.cat(condition_concat, dim=1)
        if embedding.shape[1] < self.config.hidden_size:
            embedding = nn.functional.pad(embedding, (0, self.config.hidden_size - embedding.shape[1]))
        return embedding

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.concat_padding_mask:
            padding_mask = padding_mask.squeeze(0)
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)
        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            if extra_pos_emb is not None:
                extra_pos_emb = rearrange(extra_pos_emb, 'B T H W D -> (T H W) B D')
                return x_B_T_H_W_D, [self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb]
            else:
                return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps)

        if "fps_aware" in self.pos_emb_cls:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D, fps=fps.cuda())  # [B, T, H, W, D]
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]
        return x_B_T_H_W_D, None

    def decoder_head(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        origin_shape: Tuple[int, int, int, int, int],  # [B, C, T, H, W]
        crossattn_mask: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del crossattn_emb, crossattn_mask
        B, C, T_before_patchify, H_before_patchify, W_before_patchify = origin_shape
        # TODO: (qsh 2024-09-27) notation here is wrong, should be updated!
        x_BT_HW_D = rearrange(x_B_T_H_W_D, "B T H W D -> (B T) (H W) D")
        x_BT_HW_D = self.final_layer(x_BT_HW_D, emb_B_D, adaln_lora_B_3D=adaln_lora_B_3D)
        # This is to ensure x_BT_HW_D has the correct shape because
        # when we merge T, H, W into one dimension, x_BT_HW_D has shape (B * T * H * W, 1*1, D).
        x_BT_HW_D = x_BT_HW_D.view(
            B * T_before_patchify // self.patch_temporal,
            H_before_patchify // self.patch_spatial * W_before_patchify // self.patch_spatial,
            -1,
        )
        x_B_D_T_H_W = rearrange(
            x_BT_HW_D,
            "(B T) (H W) (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            H=H_before_patchify // self.patch_spatial,
            W=W_before_patchify // self.patch_spatial,
            t=self.patch_temporal,
            B=B,
        )
        return x_B_D_T_H_W

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

        rope_emb_L_1_1_D = None
        if self.pre_process:
            x_B_T_H_W_D, rope_emb_L_1_1_D = self.prepare_embedded_sequence(x, fps=fps, padding_mask=padding_mask)
            B, T, H, W, D = x_B_T_H_W_D.shape
            # print(f'x_T_H_W_B_D.shape={x_T_H_W_B_D.shape}')
            x_S_B_D = rearrange(x_B_T_H_W_D, "B T H W D -> (T H W) B D")
            # print(f'x_S_B_D.shape={x_S_B_D.shape}')
        else:
            # intermediate stage of pipeline
            x_S_B_D = None  ### should it take encoder_hidden_states

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

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
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

        for param_name, param in self.t_embedder.named_parameters():
            weight_key = f'{prefix}t_embedder.{param_name}'
            self.tie_embeddings_weights_state_dict(param, sharded_state_dict, weight_key, weight_key)

        # for cond_name, embedder in self.additional_timestamp_embedder.items():
        #     for (param_name, param) in embedder.named_parameters():
        #         weight_key = f'{prefix}additional_t_embedder_{cond_name}.{param_name}'
        #         self.tie_embeddings_weights_state_dict(param, sharded_state_dict, weight_key, weight_key)

        return sharded_state_dict

    def tie_embeddings_weights_state_dict(
        self,
        tensor,
        sharded_state_dict: ShardedStateDict,
        output_layer_weight_key: str,
        first_stage_word_emb_key: str,
    ) -> None:
        """Ties the embedding and output weights in a given sharded state dict.

        Args:
            sharded_state_dict (ShardedStateDict): state dict with the weight to tie
            output_layer_weight_key (str): key of the output layer weight in the state dict.
                This entry will be replaced with a tied version
            first_stage_word_emb_key (str): this must be the same as the
                ShardedTensor.key of the first stage word embeddings.

        Returns: None, acts in-place
        """
        if self.pre_process and parallel_state.get_tensor_model_parallel_rank() == 0:
            # Output layer is equivalent to the embedding already
            return

        # Replace the default output layer with a one sharing the weights with the embedding
        del sharded_state_dict[output_layer_weight_key]
        last_stage_word_emb_replica_id = (
            0,  # copy of first stage embedding
            parallel_state.get_tensor_model_parallel_rank()
            + parallel_state.get_pipeline_model_parallel_rank()
            * parallel_state.get_pipeline_model_parallel_world_size(),
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )

        sharded_state_dict[output_layer_weight_key] = make_sharded_tensor_for_checkpoint(
            tensor=tensor,
            key=first_stage_word_emb_key,
            replica_id=last_stage_word_emb_replica_id,
            allow_shape_mismatch=False,
        )


class DiTCrossAttentionExtendModel7B(VisionModule):
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
        config: TransformerConfig,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        position_embedding_type: Literal["learned_absolute", "rope"] = "rope",
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[float] = None,
    ):

        super().__init__(config=config)

        self.config: TransformerConfig = config
        from megatron.core.enums import ModelType

        self.model_type = ModelType.encoder_or_decoder

        self.transformer_decoder_layer_spec = DiTLayerWithAdaLNspec()
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = True
        self.add_decoder = True
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.position_embedding_type = position_embedding_type
        self.share_embeddings_and_output_weights = False
        self.additional_timestamp_channels = False
        self.concat_padding_mask = True
        self.pos_emb_cls = 'rope3d'
        self.use_adaln_lora = True
        self.patch_spatial = 2
        self.patch_temporal = 1
        self.out_channels = 16
        self.adaln_lora_dim = 256

        # Transformer decoder
        self.decoder = TransformerBlock(
            config=self.config,
            spec=self.transformer_decoder_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=False,
        )

        self.t_embedder = nn.Sequential(
            SDXLTimesteps(self.config.hidden_size),
            SDXLTimestepEmbedding(
                self.config.hidden_size, self.config.hidden_size, use_adaln_lora=self.use_adaln_lora
            ),
        )

        if self.pre_process:
            self.in_channels = 17
            self.in_channels = self.in_channels + 1 if self.concat_padding_mask else self.in_channels
            self.legacy_patch_emb = False
            self.x_embedder = (
                PatchEmbed(
                    spatial_patch_size=self.patch_spatial,
                    temporal_patch_size=self.patch_temporal,
                    in_channels=self.in_channels,
                    out_channels=self.config.hidden_size,
                    bias=False,
                    keep_spatio=True,
                    legacy_patch_emb=self.legacy_patch_emb,
                )
                .cuda()
                .to(dtype=torch.bfloat16)
            )

            self.max_img_h = 240
            self.max_img_w = 240
            self.max_frames = 128
            self.min_fps = 1
            self.max_fps = 30
            self.pos_emb_learnable = False
            self.pos_emb_interpolation = 'crop'
            self.rope_h_extrp_ratio = 1.0
            self.rope_w_extrp_ratio = 1.0
            self.rope_t_extrp_ratio = 2.0
            self.extra_per_block_abs_pos_emb = True

            self.pos_embedder = VideoRopePosition3DEmb(
                model_channels=self.config.hidden_size,
                len_h=self.max_img_h // self.patch_spatial,
                len_w=self.max_img_w // self.patch_spatial,
                len_t=self.max_frames // self.patch_temporal,
                max_fps=self.max_fps,
                min_fps=self.min_fps,
                is_learnable=self.pos_emb_learnable,
                interpolation=self.pos_emb_interpolation,
                head_dim=self.config.hidden_size // self.config.num_attention_heads,
                h_extrp_ratio=self.rope_h_extrp_ratio,
                w_extrp_ratio=self.rope_w_extrp_ratio,
                t_extrp_ratio=self.rope_t_extrp_ratio,
            )

            if self.extra_per_block_abs_pos_emb:

                self.extra_pos_embedder = SinCosPosEmbAxis(
                    h_extrapolation_ratio=1,
                    w_extrapolation_ratio=1,
                    t_extrapolation_ratio=1,
                    model_channels=self.config.hidden_size,
                    len_h=self.max_img_h // self.patch_spatial,
                    len_w=self.max_img_w // self.patch_spatial,
                    len_t=self.max_frames // self.patch_temporal,
                    interpolation=self.pos_emb_interpolation,
                )

            if parallel_state.get_context_parallel_world_size() > 1:
                cp_group = parallel_state.get_context_parallel_group()
                self.pos_embedder.enable_context_parallel(cp_group)
                self.extra_pos_embedder.enable_context_parallel(cp_group)

        if self.post_process:
            self.final_layer = FinalLayer(
                hidden_size=self.config.hidden_size,
                spatial_patch_size=self.patch_spatial,
                temporal_patch_size=self.patch_temporal,
                out_channels=self.out_channels,
                use_adaln_lora=self.use_adaln_lora,
                adaln_lora_dim=self.adaln_lora_dim,
            )

        self.build_additional_timestamp_embedder()
        # self.affline_norm = RMSNorm(self.config.hidden_size)
        import transformer_engine as te

        self.affline_norm = te.pytorch.RMSNorm(self.config.hidden_size, eps=1e-6)
        self.logvar = nn.Sequential(
            FourierFeatures(num_channels=128, normalize=True), torch.nn.Linear(128, 1, bias=False)
        )

    def build_additional_timestamp_embedder(self):
        if self.additional_timestamp_channels:
            self.additional_timestamp_channels = dict(fps=256, h=256, w=256, org_h=256, org_w=256)
            self.additional_timestamp_embedder = nn.ModuleDict()
            for cond_name, cond_emb_channels in self.additional_timestamp_channels.items():
                print(f"Building additional timestamp embedder for {cond_name} with {cond_emb_channels} channels")
                self.additional_timestamp_embedder[cond_name] = nn.Sequential(
                    SDXLTimesteps(cond_emb_channels),
                    SDXLTimestepEmbedding(cond_emb_channels, cond_emb_channels),
                )

    def prepare_additional_timestamp_embedder(self, **kwargs):
        condition_concat = []
        for cond_name, embedder in self.additional_timestamp_embedder.items():
            condition_concat.append(embedder(kwargs[cond_name]))
        embedding = torch.cat(condition_concat, dim=1)
        if embedding.shape[1] < self.config.hidden_size:
            embedding = nn.functional.pad(embedding, (0, self.config.hidden_size - embedding.shape[1]))
        return embedding

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.concat_padding_mask:
            padding_mask = padding_mask.squeeze(0)
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)
        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            if extra_pos_emb is not None:
                extra_pos_emb = rearrange(extra_pos_emb, 'B T H W D -> (T H W) B D')
                return x_B_T_H_W_D, [self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb]
            else:
                return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps)

        if "fps_aware" in self.pos_emb_cls:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D, fps=fps.cuda())  # [B, T, H, W, D]
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]
        return x_B_T_H_W_D, None

    def decoder_head(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        origin_shape: Tuple[int, int, int, int, int],  # [B, C, T, H, W]
        crossattn_mask: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del crossattn_emb, crossattn_mask
        B, C, T_before_patchify, H_before_patchify, W_before_patchify = origin_shape
        # TODO: (qsh 2024-09-27) notation here is wrong, should be updated!
        x_BT_HW_D = rearrange(x_B_T_H_W_D, "B T H W D -> (B T) (H W) D")
        x_BT_HW_D = self.final_layer(x_BT_HW_D, emb_B_D, adaln_lora_B_3D=adaln_lora_B_3D)
        # This is to ensure x_BT_HW_D has the correct shape because
        # when we merge T, H, W into one dimension, x_BT_HW_D has shape (B * T * H * W, 1*1, D).
        x_BT_HW_D = x_BT_HW_D.view(
            B * T_before_patchify // self.patch_temporal,
            H_before_patchify // self.patch_spatial * W_before_patchify // self.patch_spatial,
            -1,
        )
        x_B_D_T_H_W = rearrange(
            x_BT_HW_D,
            "(B T) (H W) (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            H=H_before_patchify // self.patch_spatial,
            W=W_before_patchify // self.patch_spatial,
            t=self.patch_temporal,
            B=B,
        )
        return x_B_D_T_H_W

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        crossattn_emb: Tensor,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        pos_ids: Tensor = None,
        condition_video_input_mask: Optional[torch.Tensor] = None,
        condition_video_pose: Optional[torch.Tensor] = None,
        x_ctrl: Optional[Dict[str, torch.Tensor]] = None,
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

        input_list = [x, condition_video_input_mask]
        if condition_video_pose is not None:
            if condition_video_pose.shape[2] > T:
                condition_video_pose = condition_video_pose[:, :, :T, :, :].contiguous()
            input_list.append(condition_video_pose)
        x = torch.cat(
            input_list,
            dim=1,
        )

        rope_emb_L_1_1_D = None
        if self.pre_process:
            x_B_T_H_W_D, rope_emb_L_1_1_D = self.prepare_embedded_sequence(x, fps=fps, padding_mask=padding_mask)
            B, T, H, W, D = x_B_T_H_W_D.shape
            # print(f'x_T_H_W_B_D.shape={x_T_H_W_B_D.shape}')
            x_S_B_D = rearrange(x_B_T_H_W_D, "B T H W D -> (T H W) B D")
            # print(f'x_S_B_D.shape={x_S_B_D.shape}')
        else:
            # intermediate stage of pipeline
            x_S_B_D = None  ### should it take encoder_hidden_states

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
        for idx, block in enumerate(self.decoder.layers):
            name = f"block{idx}"
            x_S_B_D, _ = block(
                x_S_B_D,
                affline_emb_B_D,
                crossattn_emb,
                None,
                rotary_pos_emb=rope_emb_L_1_1_D[0],
                packed_seq_params=packed_seq_params,
            )
            if x_ctrl is not None and name in x_ctrl:
                # x_ctrl_S_B_D = rearrange(x_ctrl[name], "A B C D E -> (A B C) D E")
                x_ctrl_S_B_D = x_ctrl[name]
                x_S_B_D = x_S_B_D + x_ctrl_S_B_D
        # Return if not post_process
        if not self.post_process:
            return x_S_B_D

        if self.config.sequence_parallel:
            x_S_B_D = tensor_parallel.gather_from_sequence_parallel_region(x_S_B_D)

        x_B_T_H_W_D = rearrange(x_S_B_D, "(T H W) B D -> B T H W D", B=B, T=T, H=H, W=W, D=D)
        x_B_T_H_W_D = self.decoder_head(x_B_T_H_W_D, affline_emb_B_D, None, original_shape, None, adaln_lora_B_3D)

        return x_B_T_H_W_D

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
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

        # Allow for shape mismatch in camera control spatial embedder for fine-tuning
        if hasattr(self.config, 'model_name'):
            if 'camera_ctrl' in self.config.model_name:
                sharded_state_dict['module.x_embedder.proj.1.weight'].allow_shape_mismatch = True

        for param_name, param in self.t_embedder.named_parameters():
            weight_key = f'{prefix}t_embedder.{param_name}'
            self.tie_embeddings_weights_state_dict(param, sharded_state_dict, weight_key, weight_key)

        if self.additional_timestamp_channels:
            for cond_name, embedder in self.additional_timestamp_embedder.items():
                for param_name, param in embedder.named_parameters():
                    weight_key = f'{prefix}additional_t_embedder_{cond_name}.{param_name}'
                    self.tie_embeddings_weights_state_dict(param, sharded_state_dict, weight_key, weight_key)

        return sharded_state_dict

    def tie_embeddings_weights_state_dict(
        self,
        tensor,
        sharded_state_dict: ShardedStateDict,
        output_layer_weight_key: str,
        first_stage_word_emb_key: str,
    ) -> None:
        """Ties the embedding and output weights in a given sharded state dict.

        Args:
            sharded_state_dict (ShardedStateDict): state dict with the weight to tie
            output_layer_weight_key (str): key of the output layer weight in the state dict.
                This entry will be replaced with a tied version
            first_stage_word_emb_key (str): this must be the same as the
                ShardedTensor.key of the first stage word embeddings.

        Returns: None, acts in-place
        """
        if self.pre_process and parallel_state.get_tensor_model_parallel_rank() == 0:
            # Output layer is equivalent to the embedding already
            return

        # Replace the default output layer with a one sharing the weights with the embedding
        del sharded_state_dict[output_layer_weight_key]
        last_stage_word_emb_replica_id = (
            0,  # copy of first stage embedding
            parallel_state.get_tensor_model_parallel_rank()
            + parallel_state.get_pipeline_model_parallel_rank()
            * parallel_state.get_pipeline_model_parallel_world_size(),
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )

        sharded_state_dict[output_layer_weight_key] = make_sharded_tensor_for_checkpoint(
            tensor=tensor,
            key=first_stage_word_emb_key,
            replica_id=last_stage_word_emb_replica_id,
            allow_shape_mismatch=False,
        )


class DiTCrossAttentionActionExtendModel7B(DiTCrossAttentionExtendModel7B):

    def __init__(
        self,
        config: TransformerConfig,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        position_embedding_type: Literal["learned_absolute", "rope"] = "rope",
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[float] = None,
    ):
        # Initialize base V2W model.
        super().__init__(
            config,
            pre_process,
            post_process,
            fp16_lm_cross_entropy,
            parallel_output,
            position_embedding_type,
            rotary_percent,
            seq_len_interpolation_factor,
        )

        self.action_mlp = ActionControlTorchMlp(
            self.config.action_emb_dim, hidden_features=config.hidden_size, out_features=config.hidden_size
        )
        self.action_mlp_3d = ActionControlTorchMlp(
            self.config.action_emb_dim,
            hidden_features=config.hidden_size,
            out_features=config.hidden_size,
            action_3d=True,
        )

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        crossattn_emb: Tensor,
        action_control_condition: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        pos_ids: Tensor = None,
        condition_video_input_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass for the action-control V2W DiT model.

        Args:
            x (Tensor): Tokenized video input of shape (B, C, T, H, W).
            timestemps (Tensor): Timestep input tensor.
            crossattn_emb (Tensor): Text embedding input for the cross-attention block.
            action_control_condition (Tensor): Action control vector of shape (B, A),
                where A is the action control dimensionality, e.g. A=7 for the bridge dataset.

        Returns:
            Tensor: loss / noise prediction Tensor
        """
        original_shape = x.shape
        B, C, T, H, W = original_shape

        fps = kwargs.get('fps', None)
        if len(fps.shape) > 1:
            fps = fps.squeeze(0)
        padding_mask = kwargs.get('padding_mask', None)
        image_size = kwargs.get('image_size', None)

        input_list = [x, condition_video_input_mask]
        x = torch.cat(
            input_list,
            dim=1,
        )

        rope_emb_L_1_1_D = None
        if self.pre_process:
            x_B_T_H_W_D, rope_emb_L_1_1_D = self.prepare_embedded_sequence(x, fps=fps, padding_mask=padding_mask)
            B, T, H, W, D = x_B_T_H_W_D.shape
            x_S_B_D = rearrange(x_B_T_H_W_D, "B T H W D -> (T H W) B D")
        else:
            # intermediate stage of pipeline
            x_S_B_D = None  ### should it take encoder_hidden_states

        _, _, D = x_S_B_D.shape

        # logging affline scale information
        affline_scale_log_info = {}

        # Compute affine embeddings from the timesteps.
        timesteps_B_D, adaln_lora_B_3D = self.t_embedder(timesteps.flatten())
        affline_emb_B_D = timesteps_B_D
        affline_scale_log_info["timesteps_B_D"] = timesteps_B_D.detach()

        # Compute the action embedding MLP.
        if action_control_condition is not None:
            # Timestamp Embedding
            action_embed = self.action_mlp(action_control_condition)
            # Adaptive Layer Norm Embedding
            action_embed_3d = self.action_mlp_3d(action_control_condition)
            # Add action conditioning to affine embedding.
            affline_emb_B_D = affline_emb_B_D + action_embed
            if adaln_lora_B_3D is not None:
                adaln_lora_B_3D = adaln_lora_B_3D + action_embed_3d

        # Setup additional channels and aggregate into affline_emb_B_D.
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

        # Run the TransformerBlock cross-attention decoder. attention_mask is re-purposed to pass in the affine embedding.
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

        # Returns (B, T, H, W, D) video prediction.
        return x_B_T_H_W_D


class DiTControl7B(DiTCrossAttentionExtendModel7B):
    """
    DiTControlNet builds on top of DiTCrossAttentionExtendModel7B and mirrors the structure of repo1's ControlNet.

    It creates a control branch (self.ctrl_net) that processes an encoded control input ("hint") and then
    uses a condition indicator mask to blend the control branch output with the original input.

    The methods get_data_and_condition and encode_latent are assumed to be provided externally.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_ctrl_branch = kwargs.pop("dropout_ctrl_branch", 0.5)
        hint_channels = kwargs.pop("hint_channels", 128)
        num_control_blocks = kwargs.pop("num_control_blocks", 3)
        num_blocks = self.config.num_layers

        if num_control_blocks is not None:
            assert num_control_blocks > 0 and num_control_blocks <= num_blocks
            kwargs["layer_mask"] = [False] * num_control_blocks + [True] * (num_blocks - num_control_blocks)
        self.random_drop_control_blocks = kwargs.pop("random_drop_control_blocks", False)
        model_channels = self.config.hidden_size
        self.model_channels = model_channels
        layer_mask = kwargs.get("layer_mask", None)
        layer_mask = [False] * num_blocks if layer_mask is None else layer_mask
        self.layer_mask = layer_mask
        self.hint_channels = hint_channels
        self.build_hint_patch_embed()
        hint_nf = [16, 16, 32, 32, 96, 96, 256]
        nonlinearity = nn.SiLU()
        input_hint_block = [nn.Linear(model_channels, hint_nf[0]), nonlinearity]
        for i in range(len(hint_nf) - 1):
            input_hint_block += [nn.Linear(hint_nf[i], hint_nf[i + 1]), nonlinearity]
        self.input_hint_block = nn.Sequential(*input_hint_block)
        # Initialize weights
        self.zero_blocks = nn.ModuleDict()
        for idx in range(num_blocks):
            if layer_mask[idx]:
                continue
            blk = nn.Linear(model_channels, model_channels)
            self.zero_blocks[f"block{idx}"] = blk
        blk = nn.Linear(hint_nf[-1], model_channels)
        self.input_hint_block.append(blk)

    def build_hint_patch_embed(self):
        concat_padding_mask, in_channels, patch_spatial, patch_temporal, model_channels = (
            self.concat_padding_mask,
            self.hint_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
        )
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder2 = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
            bias=False,
            keep_spatio=True,
            legacy_patch_emb=False,
        )

    def prepare_hint_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[2], 1, 1)],
                dim=1,
            )

        x_B_T_H_W_D = self.x_embedder2(x_B_C_T_H_W)

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps)

        if "fps_aware" in self.pos_emb_cls:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D, fps=fps)  # [B, T, H, W, D]
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]
        return x_B_T_H_W_D, None

    def encode_hint(
        self,
        hint: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
    ) -> torch.Tensor:
        assert (
            hint.size(1) <= self.hint_channels
        ), f"Expected hint channels <= {self.hint_channels}, got {hint.size(1)}"
        if hint.size(1) < self.hint_channels:
            padding_shape = list(hint.shape)
            padding_shape[1] = self.hint_channels - hint.size(1)
            hint = torch.cat([hint, torch.zeros(*padding_shape, dtype=hint.dtype, device=hint.device)], dim=1)
        assert isinstance(
            data_type, DataType
        ), f"Expected DataType, got {type(data_type)}. We need discuss this flag later."

        hint_B_T_H_W_D, _ = self.prepare_hint_embedded_sequence(hint, fps=fps, padding_mask=padding_mask)

        hint = hint_B_T_H_W_D

        guided_hint = self.input_hint_block(hint)
        return guided_hint

    def encode_latent(self, data_batch: dict, tokenizer, cond_mask: list = []) -> torch.Tensor:
        x = data_batch[data_batch["hint_key"]]
        latent = []
        # control input goes through tokenizer, which always takes 3-input channels
        num_conditions = x.size(1) // 3  # input conditions were concatenated along channel dimension
        if num_conditions > 1 and self.config.hint_dropout_rate > 0:
            if torch.is_grad_enabled():  # during training, randomly dropout some conditions
                cond_mask = torch.rand(num_conditions) > self.config.hint_dropout_rate
                if not cond_mask.any():  # make sure at least one condition is present
                    cond_mask = [True] * num_conditions
            elif not cond_mask:  # during inference, use hint_mask to indicate which conditions are used
                cond_mask = self.config.hint_mask
        else:
            cond_mask = [True] * num_conditions
        for idx in range(0, x.size(1), 3):
            x_rgb = x[:, idx : idx + 3]
            if not cond_mask[idx // 3]:  # if the condition is not selected, replace with a black image
                x_rgb = torch.zeros_like(x_rgb)
            latent.append(tokenizer.encode(x_rgb) * self.sigma_data)
        latent = torch.cat(latent, dim=1)
        return latent

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        crossattn_emb: Tensor,
        data_type: Optional[DataType] = DataType.VIDEO,
        hint_key: Optional[str] = None,
        base_model: Optional[nn.Module] = None,
        control_weight: Optional[float] = 1.0,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        pos_ids: Tensor = None,
        condition_video_input_mask: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
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
        x_input = x.clone()
        crossattn_emb_input = crossattn_emb
        condition_video_input_mask_input = condition_video_input_mask
        fps = kwargs.get("fps", None)
        padding_mask = kwargs.get("padding_mask", None)
        hint = kwargs.get(hint_key)
        crossattn_mask = kwargs.get("crossattn_mask", None)

        if hint is None:
            return base_model.module.module.module.forward(
                x=x_input,
                timesteps=timesteps,
                crossattn_emb=crossattn_emb_input,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                pos_ids=pos_ids,
                condition_video_input_mask=condition_video_input_mask_input,
                **kwargs,
            )
        guided_hints = self.encode_hint(hint, fps=fps, padding_mask=padding_mask, data_type=data_type)

        guided_hints = torch.chunk(guided_hints, hint.shape[0] // x.shape[0], dim=3)
        assert isinstance(
            data_type, DataType
        ), f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        B, C, T, H, W = x_input.shape
        if data_type == DataType.VIDEO:
            if condition_video_input_mask is not None:
                input_list = [x, condition_video_input_mask]
                x = torch.cat(input_list, dim=1)

        elif data_type == DataType.IMAGE:
            # For image, we dont have condition_video_input_mask, or condition_video_pose
            # We need to add the extra channel for video condition mask
            padding_channels = self.in_channels - x.shape[1]
            x = torch.cat([x, torch.zeros((B, padding_channels, T, H, W), dtype=x.dtype, device=x.device)], dim=1)
        else:
            assert x.shape[1] == self.in_channels, f"Expected {self.in_channels} channels, got {x.shape[1]}"
        x_B_T_H_W_D, rope_emb_L_1_1_D = self.prepare_embedded_sequence(x, fps=fps, padding_mask=padding_mask)
        # logging affline scale information
        affline_scale_log_info = {}

        timesteps_B_D, adaln_lora_B_3D = self.t_embedder(timesteps.flatten())
        affline_emb_B_D = timesteps_B_D
        affline_scale_log_info["timesteps_B_D"] = timesteps_B_D.detach()

        if scalar_feature is not None:
            raise NotImplementedError("Scalar feature is not implemented yet.")
            timesteps_B_D = timesteps_B_D + scalar_feature.mean(dim=1)

        affline_scale_log_info["affline_emb_B_D"] = affline_emb_B_D.detach()
        affline_emb_B_D = self.affline_norm(affline_emb_B_D)

        # for logging purpose
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = affline_emb_B_D
        self.crossattn_emb = crossattn_emb
        self.crossattn_mask = crossattn_mask

        # if self.use_cross_attn_mask:
        #     crossattn_mask = crossattn_mask[:, None, None, :].to(dtype=torch.bool)  # [B, 1, 1, length]
        # else:
        crossattn_mask = None

        x_S_B_D = rearrange(x_B_T_H_W_D, "B T H W D -> (T H W) B D")
        x = x_S_B_D
        crossattn_emb = rearrange(crossattn_emb, 'B S D -> S B D')
        outs = {}

        # If also training base model, sometimes drop the controlnet branch to only train base branch.
        # This is to prevent the network become dependent on controlnet branch and make control weight useless.
        is_training = torch.is_grad_enabled()
        is_training_base_model = any(p.requires_grad for p in base_model.parameters())
        if is_training and is_training_base_model:
            coin_flip = torch.rand(B).to(x.device) > self.dropout_ctrl_branch  # prob for only training base model
            coin_flip = coin_flip[:, None, None, None, None]
        else:
            coin_flip = 1

        num_control_blocks = self.layer_mask.index(True)
        if self.random_drop_control_blocks:
            if is_training:  # Use a random number of layers during training.
                num_layers_to_use = np.random.randint(num_control_blocks) + 1
            elif num_layers_to_use == -1:  # Evaluate using all the layers.
                num_layers_to_use = num_control_blocks
            else:  # Use the specified number of layers during inference.
                pass
        else:  # Use all of the layers.
            num_layers_to_use = num_control_blocks

        if isinstance(control_weight, torch.Tensor) and control_weight.shape[0] > 1:
            pass
        else:
            control_weight = [control_weight] * len(guided_hints)

        packed_seq_params = {
            'adaln_lora_B_3D': adaln_lora_B_3D.detach(),
            'extra_pos_emb': rope_emb_L_1_1_D[1].detach(),
        }
        for i, guided_hint in enumerate(guided_hints):
            for idx, block in enumerate(self.decoder.layers):
                name = f"block{idx}"
                x, _ = block(
                    x,
                    affline_emb_B_D,
                    crossattn_emb,
                    None,
                    rotary_pos_emb=rope_emb_L_1_1_D[0],
                    packed_seq_params=packed_seq_params,
                )
                if guided_hint is not None:
                    gh_S_B_D = rearrange(guided_hint, 'B T H W D -> (T H W) B D')
                    x = x + gh_S_B_D
                    gh_S_B_D = None
                    guided_hint = None

                gate = idx < num_control_blocks
                # x_B_T_H_W_D = rearrange(x, '(T H W) B D -> B T H W D', T=T, H=H, W=W)
                if gate:
                    hint_val = self.zero_blocks[name](x) * control_weight[i] * coin_flip * gate
                    if name not in outs:
                        outs[name] = hint_val
                    else:
                        outs[name] += hint_val
        output = base_model.module.module.module.forward(
            x=x_input,
            x_ctrl=outs,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb_input,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            pos_ids=pos_ids,
            condition_video_input_mask=condition_video_input_mask_input,
            **kwargs,
        )
        return output


# flake8: noqa: E741
