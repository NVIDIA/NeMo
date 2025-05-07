# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
