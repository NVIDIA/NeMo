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


def make_mcore_dist_opt_wrapper(optim):

    class McoreDistributedOptimizerWrapper(type(optim), torch.optim.Optimizer):
        """
        A wrapper for Mcore distributed optimizer.

        Arguments:
            optim: distributed optimizer from Megatron core.
        """

        def sharded_state_dict(
            self, model_sharded_state_dict, optimizer_state_dict=None, is_loading=False, dist_ckpt_parallel_save=False
        ):
            sharding_type = 'fully_sharded_model_space' if dist_ckpt_parallel_save else 'dp_zero_gather_scatter'
            return self.sharded_state_dict(
                model_sharded_state_dict, is_loading=is_loading, sharding_type=sharding_type
            )

        def step(self, closure):
            """Clip gradients (if needed) and step the base optimizer.
            Always return successful since there is no overflow."""
            # Apply closure
            loss = None
            if closure is not None:
                loss = closure()

            # return unused update_successful, grad_norm, num_zeros_in_grad
            super().step()

            return loss

    optim.__class__ = McoreDistributedOptimizerWrapper
    return optim
