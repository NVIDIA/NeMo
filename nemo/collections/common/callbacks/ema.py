# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import contextlib
import copy
import threading

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class EMA(Callback):
    """
    Implements Exponential Moving Averaging (EMA).

    When training a model, this callback will maintain moving averages of the trained parameters.
    When evaluating, we use the moving averages copy of the trained parameters.
    When saving, we save an additional set of parameters with the prefix `ema`.

    Args:
        decay: The exponential decay used when calculating the moving average. Has to be between 0-1.
        validate_original_weights: Validate the original weights, as apposed to the EMA weights.
        cpu_offload: Offload weights to CPU.
    """

    def __init__(self, decay: float, validate_original_weights: bool = False, cpu_offload: bool = False):
        if not (0 <= decay <= 1):
            raise MisconfigurationException("EMA decay value must be between 0 and 1")
        self.decay = decay
        self.validate_original_weights = validate_original_weights
        self.cpu_offload = cpu_offload

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        device = pl_module.device if not self.cpu_offload else torch.device('cpu')
        trainer.optimizers = [EMAOptimizer(optim, device=device, decay=self.decay) for optim in trainer.optimizers]

    def swap_model_weights(self, trainer: "pl.Trainer"):
        for optimizer in trainer.optimizers:
            assert isinstance(optimizer, EMAOptimizer)
            optimizer.switch_main_parameter_weights()

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.validate_original_weights:
            self.swap_model_weights(trainer)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.validate_original_weights:
            self.swap_model_weights(trainer)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.validate_original_weights:
            self.swap_model_weights(trainer)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.validate_original_weights:
            self.swap_model_weights(trainer)


@torch.no_grad()
def ema_update(ema_model_tuple, current_model_tuple, decay):
    torch._foreach_mul_(ema_model_tuple, decay)
    torch._foreach_add_(
        ema_model_tuple, current_model_tuple, alpha=(1.0 - decay),
    )


def run_ema_update_cpu(ema_model_tuple, current_model_tuple, decay, pre_sync_stream=None):
    if pre_sync_stream is not None:
        pre_sync_stream.synchronize()

    ema_update(ema_model_tuple, current_model_tuple, decay)


class EMAOptimizer(torch.optim.Optimizer):
    r"""
    EMAOptimizer is a wrapper for torch.optim.Optimizer that computes
    Exponential Moving Average of parameters registered in the optimizer.

    EMA parameters are automatically updated after every step of the optimizer
    with the following formula:

        ema_weight = decay * ema_weight + (1 - decay) * training_weight

    To access EMA parameters, use ``swap_ema_weights()`` context manager to
    perform a temporary in-place swap of regular parameters with EMA
    parameters.

    Notes:
        - EMAOptimizer is not compatible with APEX AMP O2.

    Args:
        optimizer (torch.optim.Optimizer): optimizer to wrap
        device (torch.device): device for EMA parameters
        decay (float): decay factor

    Returns:
        returns an instance of torch.optim.Optimizer that computes EMA of
        parameters

    Example:
        model = Model().to(device)
        opt = torch.optim.Adam(model.parameters())

        opt = EMAOptimizer(opt, device, 0.9999)

        for epoch in range(epochs):
            training_loop(model, opt)

            regular_eval_accuracy = evaluate(model)

            with opt.swap_ema_weights():
                ema_eval_accuracy = evaluate(model)
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, device: torch.device, decay: float = 0.9999,
    ):
        self.optimizer = optimizer
        self.decay = decay
        self.device = device

        self.first_iteration = True
        self.rebuild_ema_params = True
        self.stream = None
        self.thread = None

        self.ema_params = ()

    def all_parameters(self):
        return (param for group in self.param_groups for param in group['params'])

    def step(self, closure=None):
        self.join()

        if self.first_iteration:
            if any(p.is_cuda for p in self.all_parameters()):
                self.stream = torch.cuda.Stream()

            self.first_iteration = False

        if self.rebuild_ema_params:
            opt_params = list(self.all_parameters())

            self.ema_params += tuple(
                copy.deepcopy(param.data.detach()).to(self.device) for param in opt_params[len(self.ema_params) :]
            )
            self.rebuild_ema_params = False

        loss = self.optimizer.step(closure)

        self.update()
        return loss

    @torch.no_grad()
    def update(self):
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.stream):
            current_model_state = tuple(
                param.data.to(self.device, non_blocking=True) for param in self.all_parameters()
            )

            if self.device.type == 'cuda':
                ema_update(self.ema_params, current_model_state, self.decay)

        if self.device.type == 'cpu':
            self.thread = threading.Thread(
                target=run_ema_update_cpu, args=(self.ema_params, current_model_state, self.decay, self.stream,),
            )
            self.thread.start()

    def swap_tensors(self, tensor1, tensor2):
        tmp = torch.empty_like(tensor1)
        tmp.copy_(tensor1)
        tensor1.copy_(tensor2)
        tensor2.copy_(tmp)

    def switch_main_parameter_weights(self):
        self.join()

        for param, ema_param in zip(self.all_parameters(), self.ema_params):
            self.swap_tensors(param.data, ema_param)

    @contextlib.contextmanager
    def swap_ema_weights(self, enabled: bool = True):
        r"""
        A context manager to in-place swap regular parameters with EMA
        parameters.
        It swaps back to the original regular parameters on context manager
        exit.

        Args:
            enabled (bool): whether the swap should be performed
        """

        if enabled:
            self.switch_main_parameter_weights()
        try:
            yield
        finally:
            if enabled:
                self.switch_main_parameter_weights()

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def join(self):
        if self.stream is not None:
            self.stream.synchronize()

        if self.thread is not None:
            self.thread.join()

    def state_dict(self):
        self.join()

        state_dict = {
            'opt': self.optimizer.state_dict(),
            'ema': self.ema_params,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.join()

        self.optimizer.load_state_dict(state_dict['opt'])
        self.ema_params = tuple(param.to(self.device) for param in copy.deepcopy(state_dict['ema']))

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)
        self.rebuild_ema_params = True
