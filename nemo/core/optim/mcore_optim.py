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


class McoreDistributedOptimizer(torch.optim.Optimizer):
    # (TODO) add more type check and comments
    # a pure wrapper; Not even going to implement super.__init__

    def __init__(self, optim):  # add type check
        self.defaults = {}
        self.mcore_optimizer = optim
        self.param_groups = self.mcore_optimizer.param_groups
        self.state = self.mcore_optimizer.state

    def zero_grad(self, set_to_none=True):
        self.mcore_optimizer.zero_grad(set_to_none)

    def reload_model_params(self):
        self.mcore_optimizer.reload_model_params()

    def state_dict(self):
        return self.mcore_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.mcore_optimizer.load_state_dict(state_dict)

    def sharded_state_dict(self, model_sharded_state_dict, is_loading: bool = False, **kwargs):
        return self.mcore_optimizer.sharded_state_dict(model_sharded_state_dict, is_loading, **kwargs)

    def step(self, closure):
        # Apply closure
        loss = None
        if closure is not None:
            loss = closure()

        update_successful, grad_norm, num_zeros_in_grad = self.mcore_optimizer.step()
        # (TODO) add log for grad norm here

        return loss

    def save_parameter_state(self, filename: str):
        self.mcore_optimizer.save_parameter_state(filename)

    def load_parameter_state(self, filename: str):
        self.mcore_optimizer.load_parameter_state(filename)

    def finish_param_sync(self, model_index):
        self.mcore_optimizer.finish_param_sync(model_index)

    def disable_pre_hook(self):
        self.mcore_optimizer.disable_pre_hook()

    def enable_pre_hook(self):
        self.mcore_optimizer.enable_pre_hook()
