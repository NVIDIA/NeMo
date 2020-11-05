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

from typing import Dict

import torch
from pytorch_lightning.metrics import Metric
from torch.distributions.categorical import Categorical

from nemo.utils import logging

__all__ = ['Perplexity']


class Perplexity(Metric):
    def __init__(
        self,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        validate_args=True
    ):
        super().__init__(
            compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step, process_group=process_group
        )
        self.validate_args = validate_args
        self.add_state('perplexities_sum', torch.tensor(.0, dtype=torch.float64), dist_reduce_fx='sum')
        self.add_state('num_distributions', torch.tensor(0, dtype=torch.int64), dist_reduce_fx='sum')

    def update(self, probs=None, logits=None):
        d = Categorical(probs, logits, validate_args=self.validate_args)
        ppl = d.perplexity()
        self.num_distributions += ppl.numel()
        self.perplexities_sum += ppl.sum()

    def compute(self):
        if self.num_distributions.eq(0):
            return None
        return self.perplexities_sums / self.num_distributions
