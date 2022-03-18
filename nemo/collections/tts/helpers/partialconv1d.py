###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class PartialConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):

        self.multi_channel = False
        self.return_mask = False
        super(PartialConv1d, self).__init__(*args, **kwargs)

        self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0])
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2]

        self.last_size = (None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    @torch.jit.ignore
    def forward(self, input: torch.Tensor, mask_in: Tuple[int, int, int] = None):
        assert len(input.shape) == 3
        # if a mask is input, or tensor shape changed, update mask ratio
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)
            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)
                if mask_in is None:
                    mask = torch.ones(1, 1, input.data.shape[2]).to(input)
                else:
                    mask = mask_in
                self.update_mask = F.conv1d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=1,
                )
                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-6)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)
        raw_out = super(PartialConv1d, self).forward(torch.mul(input, mask) if mask_in is not None else input)
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output
