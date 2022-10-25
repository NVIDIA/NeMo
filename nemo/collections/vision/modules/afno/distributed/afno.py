from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from numpy.lib.arraypad import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch import Tensor
from torch.nn.modules.container import Sequential
from torch.utils.checkpoint import checkpoint_sequential
from typing import Optional, Dict, List, Tuple
import math

# distributed stuff
import torch.distributed as dist

from modulus.distributed.manager import DistributedManager

from modulus.key import Key
from modulus.models.arch import Arch
from modulus.models.afno.distributed.mappings import copy_to_matmul_parallel_region
from modulus.models.afno.distributed.mappings import (
    scatter_to_matmul_parallel_region,
    gather_from_matmul_parallel_region,
)
from modulus.models.afno.distributed.layers import trunc_normal_, DropPath
from modulus.models.afno.distributed.layers import (
    DistributedPatchEmbed,
    DistributedMLP,
    DistributedAFNO2D,
)

import logging

logger = logging.getLogger(__name__)


class DistributedBlock(nn.Module):
    def __init__(
        self,
        h,
        w,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        double_skip=True,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        input_is_matmul_parallel=False,
        output_is_matmul_parallel=False,
    ):
        super(DistributedBlock, self).__init__()

        # model parallelism
        # matmul parallelism
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel

        # norm layer
        self.norm1 = norm_layer((h, w))

        # filter
        self.filter = DistributedAFNO2D(
            dim,
            num_blocks,
            sparsity_threshold,
            hard_thresholding_fraction,
            input_is_matmul_parallel=True,
            output_is_matmul_parallel=True,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # norm layer
        self.norm2 = norm_layer((h, w))

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DistributedMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            input_is_matmul_parallel=True,
            output_is_matmul_parallel=True,
        )
        self.double_skip = double_skip

    def forward(self, x):

        if not self.input_is_matmul_parallel:
            x = scatter_to_matmul_parallel_region(x, dim=1)

        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual

        if not self.output_is_matmul_parallel:
            x = gather_from_matmul_parallel_region(x, dim=1)

        return x


class DistributedAFNONet(nn.Module):
    def __init__(
        self,
        img_size=(720, 1440),
        patch_size=(16, 16),
        in_chans=2,
        out_chans=2,
        embed_dim=768,
        depth=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        num_blocks=16,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        input_is_matmul_parallel=False,
        output_is_matmul_parallel=False,
    ):
        super().__init__()

        # comm sizes
        matmul_comm_size = DistributedManager().group_size("model_parallel")

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = DistributedPatchEmbed(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=embed_dim,
            input_is_matmul_parallel=self.input_is_matmul_parallel,
            output_is_matmul_parallel=True,
        )
        num_patches = self.patch_embed.num_patches

        # original: x = B, H*W, C
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # new: x = B, C, H*W
        self.embed_dim_local = self.embed_dim // matmul_comm_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim_local, num_patches))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        # add blocks
        blks = []
        for i in range(0, depth):
            input_is_matmul_parallel = True  # if i > 0 else False
            output_is_matmul_parallel = True if i < (depth - 1) else False
            blks.append(
                DistributedBlock(
                    h=self.h,
                    w=self.w,
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    num_blocks=self.num_blocks,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                    input_is_matmul_parallel=input_is_matmul_parallel,
                    output_is_matmul_parallel=output_is_matmul_parallel,
                )
            )
        self.blocks = nn.ModuleList(blks)

        # head
        if self.output_is_matmul_parallel:
            self.out_chans_local = (
                self.out_chans + matmul_comm_size - 1
            ) // matmul_comm_size
        else:
            self.out_chans_local = self.out_chans
        self.head = nn.Conv2d(
            self.embed_dim,
            self.out_chans_local * self.patch_size[0] * self.patch_size[1],
            1,
            bias=False,
        )
        self.synchronized_head = False

        # init weights
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # reshape
        x = x.reshape(B, self.embed_dim_local, self.h, self.w)

        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):

        # fw pass on features
        x = self.forward_features(x)

        # be careful if head is distributed
        if self.output_is_matmul_parallel:
            x = copy_to_matmul_parallel_region(x)
        else:
            if not self.synchronized_head:
                # If output is not model parallel, synchronize all GPUs params for head
                for param in self.head.parameters():
                    dist.broadcast(
                        param, 0, group=DistributedManager().group("model_parallel")
                    )
                self.synchronized_head = True

        x = self.head(x)

        # new: B, C, H, W
        b = x.shape[0]
        xv = x.view(b, self.patch_size[0], self.patch_size[1], -1, self.h, self.w)
        xvt = torch.permute(xv, (0, 3, 4, 1, 5, 2)).contiguous()
        x = xvt.view(
            b, -1, (self.h * self.patch_size[0]), (self.w * self.patch_size[1])
        )

        return x


class DistributedAFNOArch(Arch):
    """Distributed Adaptive Fourier neural operator (AFNO) model.

    Note
    ----
    AFNO is a model that is designed for 2D images only.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list. The key dimension size should equal the variables channel dim.
    output_keys : List[Key]
        Output key list. The key dimension size should equal the variables channel dim.
    img_shape : Tuple[int, int]
        Input image dimensions (height, width)
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    patch_size : int, optional
        Size of image patchs, by default 16
    embed_dim : int, optional
        Embedded channel size, by default 256
    depth : int, optional
        Number of AFNO layers, by default 4
    num_blocks : int, optional
        Number of blocks in the frequency weight matrices, by default 4
    channel_parallel_inputs : bool, optional
        Are the inputs sharded along the channel dimension, by default False
    channel_parallel_outputs : bool, optional
        Should the outputs be sharded along the channel dimension, by default False

    Variable Shape
    --------------
    - Input variable tensor shape: :math:`[N, size, H, W]`
    - Output variable tensor shape: :math:`[N, size, H, W]`

    Example
    -------
    >>> afno = .afno.DistributedAFNOArch([Key("x", size=2)], [Key("y", size=2)], (64, 64))
    >>> model = afno.make_node()
    >>> input = {"x": torch.randn(20, 2, 64, 64)}
    >>> output = model(input)
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        img_shape: Tuple[int, int],
        detach_keys: List[Key] = [],
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 4,
        num_blocks: int = 4,
        channel_parallel_inputs: bool = False,
        channel_parallel_outputs: bool = False,
    ) -> None:
        super().__init__(input_keys=input_keys, output_keys=output_keys)

        self.input_keys = input_keys
        self.output_keys = output_keys
        self.detach_keys = detach_keys

        self.input_key_dict = {var.name: var.size for var in self.input_keys}
        self.output_key_dict = {var.name: var.size for var in self.output_keys}

        in_channels = sum(self.input_key_dict.values())
        out_channels = sum(self.output_key_dict.values())

        if DistributedManager().group("model_parallel") is None:
            raise RuntimeError(
                "Distributed AFNO needs to have model parallel group created first. Check the MODEL_PARALLEL_SIZE environment variable"
            )

        comm_size = DistributedManager().group_size("model_parallel")
        if channel_parallel_inputs:
            assert (
                in_channels % comm_size == 0
            ), "Error, in_channels needs to be divisible by model_parallel size"

        self._impl = DistributedAFNONet(
            img_size=img_shape,
            patch_size=(patch_size, patch_size),
            in_chans=in_channels,
            out_chans=out_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_blocks=num_blocks,
            input_is_matmul_parallel=False,
            output_is_matmul_parallel=False,
        )

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.prepare_input(
            in_vars,
            mask=self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=1,
            input_scales=self.input_scales,
        )
        y = self._impl(x)
        return self.prepare_output(
            y, output_var=self.output_key_dict, dim=1, output_scales=self.output_scales
        )

    def make_node(self, name: str, jit: bool = False, optimize: bool = True):
        """
        Override make_node method to automatically turn JIT off
        """
        if jit:
            logger.warning(
                "JIT compilation not supported for DistributedAFNOArch. Creating node with JIT turned off"
            )
        return super().make_node(name, False, optimize)
