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

import torch
import torch.nn.functional as F


def rnnt_logprobs_torch(x: torch.Tensor, targets: torch.Tensor, blank_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = x.device
    batch_size = x.shape[0]
    x_log_softmax = F.log_softmax(x, dim=-1)
    blank_scores = x_log_softmax[..., blank_id]
    targets = torch.cat((targets, torch.zeros(batch_size, dtype=targets.dtype, device=device).unsqueeze(1)), dim=-1)
    target_scores = torch.gather(
        x_log_softmax, dim=-1, index=targets.unsqueeze(1).expand(x.shape[:-1]).unsqueeze(-1)
    ).squeeze(-1)
    return target_scores, blank_scores
