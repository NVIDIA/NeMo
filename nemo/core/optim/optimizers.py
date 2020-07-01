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

from argparse import ArgumentParser
from functools import partial

import torch.optim as optim
from torch.optim import adadelta, adagrad, adamax, rmsprop, rprop
from torch.optim.optimizer import Optimizer

from nemo.core.optim.novograd import Novograd

__all__ = ['get_optimizer', 'register_optimizer', 'parse_optimizer_args', 'add_optimizer_args']


AVAILABLE_OPTIMIZERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'adadelta': adadelta.Adadelta,
    'adamax': adamax.Adamax,
    'adagrad': adagrad.Adagrad,
    'rmsprop': rmsprop.RMSprop,
    'rprop': rprop.Rprop,
    'novograd': Novograd,
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
