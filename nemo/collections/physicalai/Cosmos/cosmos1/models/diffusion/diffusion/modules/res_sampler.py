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
A general framework for various sampling algorithm from a diffusion model.
Impl based on
* Refined Exponential Solver (RES) in https://arxiv.org/pdf/2308.02157
* also clude other impl, DDIM, DEIS, DPM-Solver, EDM sampler.
Most of sampling algorihtm, Runge-Kutta, Multi-step, etc, can be impl in this framework by \
    adding new step function in get_runge_kutta_fn or get_multi_step_fn.
"""

import math
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import attrs
import torch

from cosmos1.models.diffusion.diffusion.functional.multi_step import get_multi_step_fn, is_multi_step_fn_supported
from cosmos1.models.diffusion.diffusion.functional.runge_kutta import get_runge_kutta_fn, is_runge_kutta_fn_supported
from cosmos1.utils.config import make_freezable

COMMON_SOLVER_OPTIONS = Literal["2ab", "2mid", "1euler"]


@make_freezable
@attrs.define(slots=False)
class SolverConfig:
    is_multi: bool = False
    rk: str = "2mid"
    multistep: str = "2ab"
    # following parameters control stochasticity, see EDM paper
    # BY default, we use deterministic with no stochasticity
    s_churn: float = 0.0
    s_t_max: float = float("inf")
    s_t_min: float = 0.05
    s_noise: float = 1.0


@make_freezable
@attrs.define(slots=False)
class SolverTimestampConfig:
    nfe: int = 50
    t_min: float = 0.002
    t_max: float = 80.0
    order: float = 7.0
    is_forward: bool = False  # whether generate forward or backward timestamps


@make_freezable
@attrs.define(slots=False)
class SamplerConfig:
    solver: SolverConfig = attrs.field(factory=SolverConfig)
    timestamps: SolverTimestampConfig = attrs.field(factory=SolverTimestampConfig)
    sample_clean: bool = True  # whether run one last step to generate clean image


def get_rev_ts(
    t_min: float, t_max: float, num_steps: int, ts_order: Union[int, float], is_forward: bool = False
) -> torch.Tensor:
    """
    Generate a sequence of reverse time steps.

    Args:
        t_min (float): The minimum time value.
        t_max (float): The maximum time value.
        num_steps (int): The number of time steps to generate.
        ts_order (Union[int, float]): The order of the time step progression.
        is_forward (bool, optional): If True, returns the sequence in forward order. Defaults to False.

    Returns:
        torch.Tensor: A tensor containing the generated time steps in reverse or forward order.

    Raises:
        ValueError: If `t_min` is not less than `t_max`.
        TypeError: If `ts_order` is not an integer or float.
    """
    if t_min >= t_max:
        raise ValueError("t_min must be less than t_max")

    if not isinstance(ts_order, (int, float)):
        raise TypeError("ts_order must be an integer or float")

    step_indices = torch.arange(num_steps + 1, dtype=torch.float64)
    time_steps = (
        t_max ** (1 / ts_order) + step_indices / num_steps * (t_min ** (1 / ts_order) - t_max ** (1 / ts_order))
    ) ** ts_order

    if is_forward:
        return time_steps.flip(dims=(0,))

    return time_steps


class Sampler(torch.nn.Module):
    def __init__(self, cfg: Optional[SamplerConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = SamplerConfig()
        self.cfg = cfg

    @torch.no_grad()
    def forward(
        self,
        x0_fn: Callable,
        x_sigma_max: torch.Tensor,
        num_steps: int = 35,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        rho: float = 7,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
        solver_option: str = "2ab",
    ) -> torch.Tensor:
        in_dtype = x_sigma_max.dtype

        def float64_x0_fn(x_B_StateShape: torch.Tensor, t_B: torch.Tensor) -> torch.Tensor:
            return x0_fn(x_B_StateShape.to(in_dtype), t_B.to(in_dtype)).to(torch.float64)

        is_multistep = is_multi_step_fn_supported(solver_option)
        is_rk = is_runge_kutta_fn_supported(solver_option)
        assert is_multistep or is_rk, f"Only support multistep or Runge-Kutta method, got {solver_option}"

        solver_cfg = SolverConfig(
            s_churn=S_churn,
            s_t_max=S_max,
            s_t_min=S_min,
            s_noise=S_noise,
            is_multi=is_multistep,
            rk=solver_option,
            multistep=solver_option,
        )
        timestamps_cfg = SolverTimestampConfig(nfe=num_steps, t_min=sigma_min, t_max=sigma_max, order=rho)
        sampler_cfg = SamplerConfig(solver=solver_cfg, timestamps=timestamps_cfg, sample_clean=True)

        return self._forward_impl(float64_x0_fn, x_sigma_max, sampler_cfg).to(in_dtype)

    @torch.no_grad()
    def _forward_impl(
        self,
        denoiser_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        noisy_input_B_StateShape: torch.Tensor,
        sampler_cfg: Optional[SamplerConfig] = None,
        callback_fns: Optional[List[Callable]] = None,
    ) -> torch.Tensor:
        """
        Internal implementation of the forward pass.

        Args:
            denoiser_fn: Function to denoise the input.
            noisy_input_B_StateShape: Input tensor with noise.
            sampler_cfg: Configuration for the sampler.
            callback_fns: List of callback functions to be called during sampling.

        Returns:
            torch.Tensor: Denoised output tensor.
        """
        sampler_cfg = self.cfg if sampler_cfg is None else sampler_cfg
        solver_order = 1 if sampler_cfg.solver.is_multi else int(sampler_cfg.solver.rk[0])
        num_timestamps = sampler_cfg.timestamps.nfe // solver_order

        sigmas_L = get_rev_ts(
            sampler_cfg.timestamps.t_min, sampler_cfg.timestamps.t_max, num_timestamps, sampler_cfg.timestamps.order
        ).to(noisy_input_B_StateShape.device)

        denoised_output = differential_equation_solver(
            denoiser_fn, sigmas_L, sampler_cfg.solver, callback_fns=callback_fns
        )(noisy_input_B_StateShape)

        if sampler_cfg.sample_clean:
            # Override denoised_output with fully denoised version
            ones = torch.ones(denoised_output.size(0), device=denoised_output.device, dtype=denoised_output.dtype)
            denoised_output = denoiser_fn(denoised_output, sigmas_L[-1] * ones)

        return denoised_output


def fori_loop(lower: int, upper: int, body_fun: Callable[[int, Any], Any], init_val: Any) -> Any:
    """
    Implements a for loop with a function.

    Args:
        lower: Lower bound of the loop (inclusive).
        upper: Upper bound of the loop (exclusive).
        body_fun: Function to be applied in each iteration.
        init_val: Initial value for the loop.

    Returns:
        The final result after all iterations.
    """
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def differential_equation_solver(
    x0_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sigmas_L: torch.Tensor,
    solver_cfg: SolverConfig,
    callback_fns: Optional[List[Callable]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates a differential equation solver function.

    Args:
        x0_fn: Function to compute x0 prediction.
        sigmas_L: Tensor of sigma values with shape [L,].
        solver_cfg: Configuration for the solver.
        callback_fns: Optional list of callback functions.

    Returns:
        A function that solves the differential equation.
    """
    num_step = len(sigmas_L) - 1

    if solver_cfg.is_multi:
        update_step_fn = get_multi_step_fn(solver_cfg.multistep)
    else:
        update_step_fn = get_runge_kutta_fn(solver_cfg.rk)

    eta = min(solver_cfg.s_churn / (num_step + 1), math.sqrt(1.2) - 1)

    def sample_fn(input_xT_B_StateShape: torch.Tensor) -> torch.Tensor:
        """
        Samples from the differential equation.

        Args:
            input_xT_B_StateShape: Input tensor with shape [B, StateShape].

        Returns:
            Output tensor with shape [B, StateShape].
        """
        ones_B = torch.ones(input_xT_B_StateShape.size(0), device=input_xT_B_StateShape.device, dtype=torch.float64)

        def step_fn(
            i_th: int, state: Tuple[torch.Tensor, Optional[List[torch.Tensor]]]
        ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
            input_x_B_StateShape, x0_preds = state
            sigma_cur_0, sigma_next_0 = sigmas_L[i_th], sigmas_L[i_th + 1]

            # algorithm 2: line 4-6
            if solver_cfg.s_t_min < sigma_cur_0 < solver_cfg.s_t_max:
                hat_sigma_cur_0 = sigma_cur_0 + eta * sigma_cur_0
                input_x_B_StateShape = input_x_B_StateShape + (
                    hat_sigma_cur_0**2 - sigma_cur_0**2
                ).sqrt() * solver_cfg.s_noise * torch.randn_like(input_x_B_StateShape)
                sigma_cur_0 = hat_sigma_cur_0

            if solver_cfg.is_multi:
                x0_pred_B_StateShape = x0_fn(input_x_B_StateShape, sigma_cur_0 * ones_B)
                output_x_B_StateShape, x0_preds = update_step_fn(
                    input_x_B_StateShape, sigma_cur_0 * ones_B, sigma_next_0 * ones_B, x0_pred_B_StateShape, x0_preds
                )
            else:
                output_x_B_StateShape, x0_preds = update_step_fn(
                    input_x_B_StateShape, sigma_cur_0 * ones_B, sigma_next_0 * ones_B, x0_fn
                )

            if callback_fns:
                for callback_fn in callback_fns:
                    callback_fn(**locals())

            return output_x_B_StateShape, x0_preds

        x_at_eps, _ = fori_loop(0, num_step, step_fn, [input_xT_B_StateShape, None])
        return x_at_eps

    return sample_fn
