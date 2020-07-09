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
import warnings
from functools import partial
from typing import Any, Dict, Optional

import torch.optim as optim
import torch.utils.data.dataloader as dataloader
from omegaconf import DictConfig
from torch.optim.lr_scheduler import _LRScheduler

from nemo import logging

__all__ = [
    'WarmupPolicy',
    'WarmupHoldPolicy',
    'SquareAnnealing',
    'CosineAnnealing',
    'WarmupAnnealing',
    'InverseSquareRootAnnealing',
    'SquareRootAnnealing',
    'PolynomialDecayAnnealing',
    'PolynomialHoldDecayAnnealing',
]


class WarmupPolicy(_LRScheduler):
    """Adds warmup kwargs and warmup logic to lr policy.
    All arguments should be passed as kwargs for clarity,
    Args:
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps
        max_steps: Total number of steps while training or `None` for
            infinite training
    """

    def __init__(self, optimizer, *, warmup_steps=None, warmup_ratio=None, max_steps=None, last_epoch=-1):
        assert not (
            warmup_steps is not None and warmup_ratio is not None
        ), "Either use particular number of step or ratio"
        assert warmup_ratio is None or max_steps is not None, "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0
        print(f'optimizer: {optimizer}')
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
            )

        step = self.last_epoch
        if step <= self.warmup_steps:
            lr_val = (step + 1) / (self.warmup_steps + 1)
            return [initial_lr * lr_val for initial_lr in self.base_lrs]

        if step > self.max_steps:
            return [0.0 for _ in self.base_lrs]

        return self._get_lr(step)

    def _get_lr(self, step):
        """Simple const lr policy"""
        return self.base_lrs


class WarmupHoldPolicy(WarmupPolicy):
    """Variant of WarmupPolicy which maintains high learning rate for a defined number of steps.
    All arguments should be passed as kwargs for clarity,
    Args:
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps
        hold_steps: Number of training steps to hold the learning rate after warm up
        hold_ratio: Ratio of hold steps to total steps
        max_steps: Total number of steps while training or `None` for
            infinite training
    """

    def __init__(
        self,
        optimizer,
        *,
        warmup_steps=None,
        warmup_ratio=None,
        hold_steps=None,
        hold_ratio=None,
        max_steps=None,
        min_lr=0.0,
        last_epoch=-1,
    ):
        assert not (hold_steps is not None and hold_ratio is not None), "Either use particular number of step or ratio"
        assert hold_ratio is None or max_steps is not None, "If there is a ratio, there should be a total steps"

        self._min_lr = min_lr
        self._last_warmup_lr = 0.0

        # Necessary to duplicate as class attributes are hidden in inner class
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        if hold_steps is not None:
            self.hold_steps = hold_steps + self.warmup_steps
        elif hold_ratio is not None:
            self.hold_steps = int(hold_ratio * max_steps) + self.warmup_steps
        else:
            self.hold_steps = 0

        super().__init__(
            optimizer,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            max_steps=max_steps,
            last_epoch=last_epoch,
        )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
            )

        step = self.last_epoch

        # Warmup phase
        if step <= self.warmup_steps:
            lr_val = (step + 1) / (self.warmup_steps + 1)
            return [initial_lr * lr_val for initial_lr in self.base_lrs]

        # Hold phase
        if (step >= self.warmup_steps) and (step < self.hold_steps):
            return self.base_lrs

        if step > self.max_steps:
            return [0.0 for _ in self.base_lrs]

        return self._get_lr(step)


def _squareroot_annealing(initial_lr, step, max_steps, min_lr):
    mult = ((max_steps - step) / max_steps) ** 0.5
    out_lr = initial_lr * mult
    out_lr = max(out_lr, min_lr)
    return out_lr


def _square_annealing(initial_lr, step, max_steps, min_lr):
    mult = ((max_steps - step) / max_steps) ** 2
    out_lr = initial_lr * mult
    out_lr = max(out_lr, min_lr)
    return out_lr


def _cosine_annealing(initial_lr, step, max_steps, min_lr):
    mult = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    out_lr = (initial_lr - min_lr) * mult + min_lr
    return out_lr


def _poly_decay(initial_lr, step, decay_steps, power, min_lr, cycle):
    if cycle:
        multiplier = 1.0 if step == 0 else math.ceil(step / decay_steps)
        decay_steps *= multiplier
    else:
        step = min(step, decay_steps)
    p = step / decay_steps
    lr = (initial_lr - min_lr) * math.pow(1.0 - p, power)
    lr += min_lr
    return lr


class SquareAnnealing(WarmupPolicy):
    def __init__(self, optimizer, *, max_steps, min_lr=1e-5, last_epoch=-1, **kwargs):
        self.min_lr = min_lr

        super().__init__(optimizer=optimizer, max_steps=max_steps, last_epoch=last_epoch, **kwargs)

    def _get_lr(self, step):
        new_lrs = [
            _square_annealing(
                initial_lr=initial_lr,
                step=step - self.warmup_steps,
                max_steps=self.max_steps - self.warmup_steps,
                min_lr=self.min_lr,
            )
            for initial_lr in self.base_lrs
        ]
        return new_lrs


class SquareRootAnnealing(WarmupPolicy):
    def __init__(self, optimizer, *, max_steps, min_lr=0, last_epoch=-1, **kwargs):
        self.min_lr = min_lr
        super().__init__(optimizer=optimizer, max_steps=max_steps, last_epoch=last_epoch, **kwargs)

    def _get_lr(self, step):
        new_lrs = [
            _squareroot_annealing(initial_lr=initial_lr, step=step, max_steps=self.max_steps, min_lr=self.min_lr,)
            for initial_lr in self.base_lrs
        ]
        return new_lrs


class CosineAnnealing(WarmupPolicy):
    def __init__(self, optimizer, *, max_steps, min_lr=0, last_epoch=-1, **kwargs):
        self.min_lr = min_lr
        super().__init__(optimizer=optimizer, max_steps=max_steps, last_epoch=last_epoch, **kwargs)

    def _get_lr(self, step):
        for initial_lr in self.base_lrs:
            if initial_lr < self.min_lr:
                raise ValueError(
                    f"{self} received an initial learning rate that " f"was lower than the minimum learning rate."
                )

        new_lrs = [
            _cosine_annealing(
                initial_lr=initial_lr,
                step=step - self.warmup_steps,
                max_steps=self.max_steps - self.warmup_steps,
                min_lr=self.min_lr,
            )
            for initial_lr in self.base_lrs
        ]
        return new_lrs


class WarmupAnnealing(WarmupPolicy):
    def __init__(self, optimizer, *, max_steps, last_epoch=-1, **kwargs):
        super().__init__(optimizer=optimizer, max_steps=max_steps, last_epoch=last_epoch, **kwargs)

    def _get_lr(self, step):
        progress = float(step / self.max_steps)
        warmup_ratio = float(self.warmup_steps / self.max_steps)

        mult = max((progress - 1.0) / (warmup_ratio - 1.0), 0.0)
        out_lr = [initial_lr * mult for initial_lr in self.base_lrs]

        return out_lr


class InverseSquareRootAnnealing(WarmupPolicy):
    def __init__(self, optimizer, *, max_steps, last_epoch=-1, **kwargs):
        super().__init__(optimizer=optimizer, max_steps=max_steps, **kwargs, last_epoch=last_epoch)

    def _get_lr(self, step):
        denom = ((step + 1) / (self.warmup_steps + 1)) ** 0.5
        out_lr = [initial_lr / denom for initial_lr in self.base_lrs]
        return out_lr


class PolynomialDecayAnnealing(WarmupPolicy):
    def __init__(self, optimizer, *, max_steps, min_lr=0.0, power=1.0, cycle=False, last_epoch=-1, **kwargs):
        self.min_lr = min_lr
        self.power = power
        self.cycle = cycle

        super().__init__(optimizer=optimizer, max_steps=max_steps, last_epoch=last_epoch, **kwargs)

    def _get_lr(self, step):
        new_lrs = [
            _poly_decay(
                initial_lr,
                step=step - self.warmup_steps,
                decay_steps=self.max_steps - self.warmup_steps,
                power=self.power,
                min_lr=self.min_lr,
                cycle=self.cycle,
            )
            for initial_lr in self.base_lrs
        ]
        return new_lrs


class PolynomialHoldDecayAnnealing(WarmupHoldPolicy):
    def __init__(self, optimizer, *, max_steps, min_lr=0.0, power=1.0, cycle=False, last_epoch=-1, **kwargs):
        self.min_lr = min_lr
        self.power = power
        self.cycle = cycle

        super().__init__(optimizer=optimizer, max_steps=max_steps, last_epoch=last_epoch, **kwargs)

    def _get_lr(self, step):
        new_lrs = [
            _poly_decay(
                initial_lr,
                step=step - self.hold_steps,
                decay_steps=self.max_steps - max(self.warmup_steps, self.hold_steps),
                power=self.power,
                min_lr=self.min_lr,
                cycle=self.cycle,
            )
            for initial_lr in self.base_lrs
        ]
        return new_lrs


AVAILABLE_SCHEDULERS = {'CosineAnnealing': CosineAnnealing}


def get_scheduler(name: str, **kwargs: Optional[Dict[str, Any]]) -> _LRScheduler:
    """
    Convenience method to obtain an _LRScheduler class and partially instantiate it with optimizer kwargs.

    Args:
        name: Name of the scheduler in the registry.
        kwargs: Optional kwargs of the scheduler used during instantiation.

    Returns:
        a partially instantiated _LRScheduler
    """
    if name not in AVAILABLE_SCHEDULERS:
        raise ValueError(
            f"Cannot resolve scheduler{name}'. Available optimizers are : " f"{AVAILABLE_SCHEDULERS.keys()}"
        )

    scheduler_cls = AVAILABLE_SCHEDULERS[name]
    scheduler = partial(scheduler_cls, **kwargs)
    return scheduler


def prepare_lr_scheduler(
    optimizer: optim.Optimizer, scheduler_config: DictConfig, train_dataloader: Optional[dataloader.DataLoader] = None,
) -> Optional[Dict[str, Any]]:
    """
    Constructs an LR Scheduler (optionally) for a given optimizer, based on a config with the following schema

    {
    ...,
    "scheduler": <a class that inherits torch.optim.lr_scheduler._LRScheduler>,
    "scheduler_args": {
        "max_steps": int, <OR> "iters_per_batch": int,
        "monitor": <metric to monitor; say "loss" or "val_loss">,
        <any kwarg to pass onto the optimizer>
      }
    }

    Args:
        optimizer: An instantiated Optimizer.
        scheduler_config: A dictionary which follows the above schema.
        train_dataloader: Optional requirement, must be passed if "iters_per_batch" is defined
            instead of "max_steps". Used to compute effective "max_steps".

    Returns:
        A dictionary containing the LR Scheduler implementation if the config was successfully parsed
        along with other parameters required by Pytorch Lightning, otherwise None.
    """
    # if 'scheduler' in scheduler_config:
    #     if 'scheduler_args' in scheduler_config:
    #         scheduler_args = scheduler_config['scheduler_args']
    #     else:
    #         raise ValueError("If `scheduler` is provided, `scheduler_args` must be provided.")

    # else:
    #     logging.info('Scheduler not initialized as no `scheduler` argument supplied to setup_optimizer()')
    #     return None

    # Get the scheduler class from the config
    # scheduler = scheduler_config['scheduler']
    scheduler = get_scheduler(scheduler_config.name)

    # Extract value to monitor in losses, if provided.
    if 'monitor' in scheduler_config:
        # monitor = scheduler_config.args.pop('monitor')
        monitor = scheduler_config.monitor
    else:
        # default to train loss
        monitor = 'loss'

    # Compute effective max_steps if iters_per_batch is provided
    if 'iters_per_batch' in scheduler_config.args:
        if train_dataloader is None:
            raise ValueError(
                'As `iters_per_batch` is provided, it is required to pass the train dataloader in order '
                'to compute effective maximum number of steps'
            )

        # iters_per_batch = scheduler_config.args.pop('iters_per_batch')
        iters_per_batch = scheduler_config.args.iters_per_batch
        num_samples = len(train_dataloader.dataset)
        batch_size = train_dataloader.batch_size
        max_steps = round(num_samples * iters_per_batch / float(batch_size))

        scheduler_config.args.max_steps = max_steps

    else:
        max_steps = scheduler_config.args.max_steps

    # Instantiate the LR schedule
    schedule = scheduler(optimizer, **scheduler_config.args)

    logging.info(
        'Scheduler "%s" will be used during training (effective maximum steps = %d)', str(schedule), max_steps
    )

    # Wrap the schedule in PTL arguments to perform stepwise computation
    # Rather than epoch level computation
    if isinstance(schedule, optim.lr_scheduler.ReduceLROnPlateau):
        reduce_lr_on_plateau = True
    else:
        reduce_lr_on_plateau = False

    schedule_dict = {
        'scheduler': schedule,
        'interval': 'step',
        'frequency': 1,
        'monitor': monitor,
        'reduce_on_plateau': reduce_lr_on_plateau,
    }
    return schedule_dict
