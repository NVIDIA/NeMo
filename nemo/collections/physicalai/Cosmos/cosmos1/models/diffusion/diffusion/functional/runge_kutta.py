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

from typing import Callable, Tuple

import torch

from cosmos1.models.diffusion.diffusion.functional.batch_ops import batch_mul


def phi1(t: torch.Tensor) -> torch.Tensor:
    """
    Compute the first order phi function: (exp(t) - 1) / t.

    Args:
        t: Input tensor.

    Returns:
        Tensor: Result of phi1 function.
    """
    input_dtype = t.dtype
    t = t.to(dtype=torch.float64)
    return (torch.expm1(t) / t).to(dtype=input_dtype)


def phi2(t: torch.Tensor) -> torch.Tensor:
    """
    Compute the second order phi function: (phi1(t) - 1) / t.

    Args:
        t: Input tensor.

    Returns:
        Tensor: Result of phi2 function.
    """
    input_dtype = t.dtype
    t = t.to(dtype=torch.float64)
    return ((phi1(t) - 1.0) / t).to(dtype=input_dtype)


def res_x0_rk2_step(
    x_s: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    x0_s: torch.Tensor,
    s1: torch.Tensor,
    x0_s1: torch.Tensor,
) -> torch.Tensor:
    """
    Perform a residual-based 2nd order Runge-Kutta step.

    Args:
        x_s: Current state tensor.
        t: Target time tensor.
        s: Current time tensor.
        x0_s: Prediction at current time.
        s1: Intermediate time tensor.
        x0_s1: Prediction at intermediate time.

    Returns:
        Tensor: Updated state tensor.

    Raises:
        AssertionError: If step size is too small.
    """
    s = -torch.log(s)
    t = -torch.log(t)
    m = -torch.log(s1)

    dt = t - s
    assert not torch.any(torch.isclose(dt, torch.zeros_like(dt), atol=1e-6)), "Step size is too small"
    assert not torch.any(torch.isclose(m - s, torch.zeros_like(dt), atol=1e-6)), "Step size is too small"

    c2 = (m - s) / dt
    phi1_val, phi2_val = phi1(-dt), phi2(-dt)

    # Handle edge case where t = s = m
    b1 = torch.nan_to_num(phi1_val - 1.0 / c2 * phi2_val, nan=0.0)
    b2 = torch.nan_to_num(1.0 / c2 * phi2_val, nan=0.0)

    return batch_mul(torch.exp(-dt), x_s) + batch_mul(dt, batch_mul(b1, x0_s) + batch_mul(b2, x0_s1))


def reg_x0_euler_step(
    x_s: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    x0_s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a regularized Euler step based on x0 prediction.

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_s: Prediction at current time.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and current prediction.
    """
    coef_x0 = (s - t) / s
    coef_xs = t / s
    return batch_mul(coef_x0, x0_s) + batch_mul(coef_xs, x_s), x0_s


def reg_eps_euler_step(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, eps_s: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a regularized Euler step based on epsilon prediction.

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        eps_s: Epsilon prediction at current time.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and current x0 prediction.
    """
    return x_s + batch_mul(eps_s, t - s), x_s + batch_mul(eps_s, 0 - s)


def rk1_euler(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a first-order Runge-Kutta (Euler) step.

    Recommended for diffusion models with guidance or model undertrained
    Usually more stable at the cost of a bit slower convergence.

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_fn: Function to compute x0 prediction.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and x0 prediction.
    """
    x0_s = x0_fn(x_s, s)
    return reg_x0_euler_step(x_s, s, t, x0_s)


def rk2_mid_stable(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a stable second-order Runge-Kutta (midpoint) step.

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_fn: Function to compute x0 prediction.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and x0 prediction.
    """
    s1 = torch.sqrt(s * t)
    x_s1, _ = rk1_euler(x_s, s, s1, x0_fn)

    x0_s1 = x0_fn(x_s1, s1)
    return reg_x0_euler_step(x_s, s, t, x0_s1)


def rk2_mid(x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_fn: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a second-order Runge-Kutta (midpoint) step.

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_fn: Function to compute x0 prediction.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and x0 prediction.
    """
    s1 = torch.sqrt(s * t)
    x_s1, x0_s = rk1_euler(x_s, s, s1, x0_fn)

    x0_s1 = x0_fn(x_s1, s1)

    return res_x0_rk2_step(x_s, t, s, x0_s, s1, x0_s1), x0_s1


def rk_2heun_naive(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a naive second-order Runge-Kutta (Heun's method) step.
    Impl based on rho-rk-deis solvers, https://github.com/qsh-zh/deis
    Recommended for diffusion models without guidance and relative large NFE

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_fn: Function to compute x0 prediction.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and current state.
    """
    x_t, x0_s = rk1_euler(x_s, s, t, x0_fn)
    eps_s = batch_mul(1.0 / s, x_t - x0_s)
    x0_t = x0_fn(x_t, t)
    eps_t = batch_mul(1.0 / t, x_t - x0_t)

    avg_eps = (eps_s + eps_t) / 2

    return reg_eps_euler_step(x_s, s, t, avg_eps)


def rk_2heun_edm(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a naive second-order Runge-Kutta (Heun's method) step.
    Impl based no EDM second order Heun method

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_fn: Function to compute x0 prediction.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and current state.
    """
    x_t, x0_s = rk1_euler(x_s, s, t, x0_fn)
    x0_t = x0_fn(x_t, t)

    avg_x0 = (x0_s + x0_t) / 2

    return reg_x0_euler_step(x_s, s, t, avg_x0)


def rk_3kutta_naive(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a naive third-order Runge-Kutta step.
    Impl based on rho-rk-deis solvers, https://github.com/qsh-zh/deis
    Recommended for diffusion models without guidance and relative large NFE

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_fn: Function to compute x0 prediction.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and current state.
    """
    c2, c3 = 0.5, 1.0
    a31, a32 = -1.0, 2.0
    b1, b2, b3 = 1.0 / 6, 4.0 / 6, 1.0 / 6

    delta = t - s

    s1 = c2 * delta + s
    s2 = c3 * delta + s
    x_s1, x0_s = rk1_euler(x_s, s, s1, x0_fn)
    eps_s = batch_mul(1.0 / s, x_s - x0_s)
    x0_s1 = x0_fn(x_s1, s1)
    eps_s1 = batch_mul(1.0 / s1, x_s1 - x0_s1)

    _eps = a31 * eps_s + a32 * eps_s1
    x_s2, _ = reg_eps_euler_step(x_s, s, s2, _eps)

    x0_s2 = x0_fn(x_s2, s2)
    eps_s2 = batch_mul(1.0 / s2, x_s2 - x0_s2)

    avg_eps = b1 * eps_s + b2 * eps_s1 + b3 * eps_s2
    return reg_eps_euler_step(x_s, s, t, avg_eps)


# key : order + name
RK_FNs = {
    "1euler": rk1_euler,
    "2mid": rk2_mid,
    "2mid_stable": rk2_mid_stable,
    "2heun_edm": rk_2heun_edm,
    "2heun_naive": rk_2heun_naive,
    "3kutta_naive": rk_3kutta_naive,
}


def get_runge_kutta_fn(name: str) -> Callable:
    """
    Get the specified Runge-Kutta function.

    Args:
        name: Name of the Runge-Kutta method.

    Returns:
        Callable: The specified Runge-Kutta function.

    Raises:
        RuntimeError: If the specified method is not supported.
    """
    if name in RK_FNs:
        return RK_FNs[name]
    methods = "\n\t".join(RK_FNs.keys())
    raise RuntimeError(f"Only support the following Runge-Kutta methods:\n\t{methods}")


def is_runge_kutta_fn_supported(name: str) -> bool:
    """
    Check if the specified Runge-Kutta function is supported.

    Args:
        name: Name of the Runge-Kutta method.

    Returns:
        bool: True if the method is supported, False otherwise.
    """
    return name in RK_FNs
