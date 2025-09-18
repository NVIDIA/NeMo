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


import torch

from nemo.collections.asr.inference.utils.constants import BIG_EPSILON, SMALL_EPSILON


def normalize_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """
    log_probs: (B, T, vocab_size) log probabilities
    """
    # Ensure log_probs are normalized
    ONE = torch.tensor(1.0, dtype=log_probs.dtype)
    if torch.allclose(log_probs[0][0].sum(), ONE, atol=BIG_EPSILON):
        # assume that softmax is already applied
        log_probs = torch.log(log_probs + SMALL_EPSILON)
    else:
        # Otherwise, check if it's already in log-softmax form
        if not torch.allclose(log_probs[0][0].exp().sum(), ONE, atol=BIG_EPSILON):
            # If it's neither prob nor log-softmax, apply log_softmax
            log_probs = torch.log_softmax(log_probs, dim=-1)
    return log_probs
