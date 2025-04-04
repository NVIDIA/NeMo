# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Impl of multistep methods to solve the ODE in the diffusion model.
"""

from typing import Callable, List, Tuple

import torch

from cosmos1.models.diffusion.diffusion.functional.runge_kutta import reg_x0_euler_step, res_x0_rk2_step


def order2_fn(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_s: torch.Tensor, x0_preds: torch.Tensor
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    impl the second order multistep method in https://arxiv.org/pdf/2308.02157
    Adams Bashforth approach!
    """
    if x0_preds:
        x0_s1, s1 = x0_preds[0]
        x_t = res_x0_rk2_step(x_s, t, s, x0_s, s1, x0_s1)
    else:
        x_t = reg_x0_euler_step(x_s, s, t, x0_s)[0]
    return x_t, [(x0_s, s)]


# key: method name, value: method function
# key: order + algorithm name
MULTISTEP_FNs = {
    "2ab": order2_fn,
}


def get_multi_step_fn(name: str) -> Callable:
    if name in MULTISTEP_FNs:
        return MULTISTEP_FNs[name]
    methods = "\n\t".join(MULTISTEP_FNs.keys())
    raise RuntimeError("Only support multistep method\n" + methods)


def is_multi_step_fn_supported(name: str) -> bool:
    """
    Check if the multistep method is supported.
    """
    return name in MULTISTEP_FNs
