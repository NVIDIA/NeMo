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

import math
from argparse import ArgumentParser
from functools import partial

import torch
import torch.optim as optim
from torch.optim import adadelta, adagrad, adamax, rmsprop, rprop
from torch.optim.optimizer import Optimizer

__all__ = ['Novograd', 'get_optimizer', 'register_optimizer', 'parse_optimizer_args', 'add_optimizer_args']


AVAILABLE_OPTIMIZERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'adadelta': adadelta.Adadelta,
    'adamax': adamax.Adamax,
    'adagrad': adagrad.Adagrad,
    'rmsprop': rmsprop.RMSprop,
    'rprop': rprop.Rprop,
}


def _boolify(s):
    if s == 'True' or s == 'true':
        return True
    if s == 'False' or s == 'false':
        return False
    raise ValueError('Not Boolean Value!')


def _autocast(value):
    # if value is itself None, dont parse
    if value is None:
        return None

    # If value is comma seperated list of items, recursively parse all items in list.
    if "," in value:
        values = value.split(',')
        values = [_autocast(value) for value in values]
        return values

    # If value is string `none` or `None`, parse as None
    if value == 'none' or 'None':
        return None

    # Try type cast and return
    for cast_type in (int, float, _boolify):
        try:
            return cast_type(value)
        except Exception:
            pass

    # All types failed, return str without casting
    return value  # str type


def _check_valid_opt_params(lr, eps, betas):
    if lr < 0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if eps < 0:
        raise ValueError(f"Invalid epsilon value: {eps}")
    if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
        raise ValueError(f"Betas have to be between 0 and 1: {betas}")


def parse_optimizer_args(optimizer_kwargs):
    kwargs = {}

    if optimizer_kwargs is None:
        return kwargs

    # If it is a pre-defined dictionary, just return its values
    if hasattr(optimizer_kwargs, 'keys'):
        return optimizer_kwargs

    # If it is key=value string list, parse all items
    for key_value in optimizer_kwargs:
        key, str_value = key_value.split('=')

        value = _autocast(str_value)
        kwargs[key] = value

    return kwargs


def add_optimizer_args(parent_parser: ArgumentParser, optimizer='adam', default_opt_args=None) -> ArgumentParser:
    """Extends existing argparse with support for optimizers.

    Args:
        parent_parser (ArgumentParser): Custom CLI parser that will be extended.
        optimizer (str): Default optimizer required.
        default_opt_args (list(str)): List of overriding arguments for the instantiated optimizer.

    Returns:
        ArgumentParser: Parser extended by Optimizers arguments.
    """
    if default_opt_args is None:
        default_opt_args = []

    parser = ArgumentParser(parents=[parent_parser], add_help=True, conflict_handler='resolve')

    parser.add_argument('--optimizer', type=str, default=optimizer, help='Name of the optimizer. Defaults to Adam.')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate of the optimizer.')
    parser.add_argument(
        '--opt_args',
        default=default_opt_args,
        nargs='+',
        type=str,
        help='Overriding arguments for the optimizer. \n'
        'Must follow the pattern : \n'
        'name=value seperated by spaces.',
    )

    return parser


def register_optimizer(name, optimizer: Optimizer):
    if name in AVAILABLE_OPTIMIZERS:
        raise ValueError(f"Cannot override pre-existing optimizers. Conflicting optimizer name = {name}")

    AVAILABLE_OPTIMIZERS[name] = optimizer


def get_optimizer(name, **kwargs):
    if name not in AVAILABLE_OPTIMIZERS:
        raise ValueError(
            f"Cannot resolve optimizer '{name}'. Available optimizers are : " f"{AVAILABLE_OPTIMIZERS.keys()}"
        )

    optimizer = AVAILABLE_OPTIMIZERS[name]
    optimizer = partial(optimizer, **kwargs)
    return optimizer


def master_params(optimizer):
    """
    Generator expression that iterates over the params owned by ``optimizer``.
    Args:
        optimizer: An optimizer previously returned from ``amp.initialize``.
    """
    for group in optimizer.param_groups:
        for p in group['params']:
            yield p


class Novograd(Optimizer):
    """Implements Novograd algorithm.
    It has been proposed  in "Stochastic Gradient Methods with Layer-wise
    Adaptive Moments for Training of Deep Networks"
    (https://arxiv.org/abs/1905.11286)
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and Beyond"
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.95, 0.98),
        eps=1e-8,
        weight_decay=0,
        grad_averaging=False,
        amsgrad=False,
        luc=False,
        luc_trust=1e-3,
        luc_eps=1e-8,
    ):
        _check_valid_opt_params(lr, eps, betas)
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, grad_averaging=grad_averaging, amsgrad=amsgrad,
        )
        self.luc = luc
        self.luc_trust = luc_trust
        self.luc_eps = luc_eps
        super(Novograd, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Novograd, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")
                amsgrad = group["amsgrad"]
                state = self.state[p]

                # State initialization
                if not state:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros([]).to(state["exp_avg"].device)
                    if amsgrad:
                        # Maintains max of all exp moving avg of squared grad
                        state["max_exp_avg_sq"] = torch.zeros([]).to(state["exp_avg"].device)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                norm = grad.norm().pow(2)

                if exp_avg_sq == 0:
                    exp_avg_sq.copy_(norm)
                else:
                    exp_avg_sq.mul_(beta2).add_(1 - beta2, norm)

                if amsgrad:
                    # Maintains max of all 2nd moment running avg till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                grad.div_(denom)
                if group["weight_decay"] != 0:
                    grad.add_(group["weight_decay"], p.data)
                if group["grad_averaging"]:
                    grad.mul_(1 - beta1)
                exp_avg.mul_(beta1).add_(grad)

                if self.luc:
                    # Clip update so that updates are less than eta*weights
                    data_norm = torch.norm(p.data)
                    grad_norm = torch.norm(exp_avg.data)
                    luc_factor = self.luc_trust * data_norm / (grad_norm + self.luc_eps)
                    luc_factor = min(luc_factor, group["lr"])
                    p.data.add_(-luc_factor, exp_avg)
                else:
                    p.data.add_(-group["lr"], exp_avg)

        return loss


# Register Novograd
register_optimizer('novograd', Novograd)
