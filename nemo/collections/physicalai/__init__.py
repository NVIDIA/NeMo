# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

"""The loss reduction modes."""

from enum import Enum

import torch


def _mean(recon: torch.Tensor) -> torch.Tensor:
    return torch.mean(recon)


def _sum_per_frame(recon: torch.Tensor) -> torch.Tensor:
    batch_size = recon.shape[0] * recon.shape[2] if recon.ndim == 5 else recon.shape[0]
    return torch.sum(recon) / batch_size


def _sum(recon: torch.Tensor) -> torch.Tensor:
    return torch.sum(recon) / recon.shape[0]


class ReduceMode(Enum):
    MEAN = "MEAN"
    SUM_PER_FRAME = "SUM_PER_FRAME"
    SUM = "SUM"

    @property
    def function(self):
        if self == ReduceMode.MEAN:
            return _mean
        elif self == ReduceMode.SUM_PER_FRAME:
            return _sum_per_frame
        elif self == ReduceMode.SUM:
            return _sum
        else:
            raise ValueError("Invalid ReduceMode")
