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

from torch.optim.lr_scheduler import _LRScheduler

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
        total_steps: Total number of steps while training or `None` for
            infinite training
    """

    def __init__(self, optimizer, *, warmup_steps=None, warmup_ratio=None, total_steps=None, last_epoch=-1):
        assert not (
            warmup_steps is not None and warmup_ratio is not None
        ), "Either use particular number of step or ratio"
        assert warmup_ratio is None or total_steps is not None, "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by inner class.
        self.total_steps = total_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * total_steps)
        else:
            self.warmup_steps = 0

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

        if step > self.total_steps:
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
        total_steps: Total number of steps while training or `None` for
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
        total_steps=None,
        min_lr=0.0,
        last_epoch=-1,
    ):
        assert not (hold_steps is not None and hold_ratio is not None), "Either use particular number of step or ratio"
        assert hold_ratio is None or total_steps is not None, "If there is a ratio, there should be a total steps"

        self._min_lr = min_lr
        self._last_warmup_lr = 0.0

        # Necessary to duplicate as class attributes are hidden in inner class
        self.total_steps = total_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * total_steps)
        else:
            self.warmup_steps = 0

        if hold_steps is not None:
            self.hold_steps = hold_steps + self.warmup_steps
        elif hold_ratio is not None:
            self.hold_steps = int(hold_ratio * total_steps) + self.warmup_steps
        else:
            self.hold_steps = 0

        super().__init__(
            optimizer,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            total_steps=total_steps,
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

        if step > self.total_steps:
            return [0.0 for _ in self.base_lrs]

        return self._get_lr(step)


def _squareroot_annealing(initial_lr, step, total_steps, min_lr):
    mult = ((total_steps - step) / total_steps) ** 0.5
    out_lr = initial_lr * mult
    out_lr = max(out_lr, min_lr)
    return out_lr


def _square_annealing(initial_lr, step, total_steps, min_lr):
    mult = ((total_steps - step) / total_steps) ** 2
    out_lr = initial_lr * mult
    out_lr = max(out_lr, min_lr)
    return out_lr


def _cosine_annealing(initial_lr, step, total_steps, min_lr):
    mult = 0.5 * (1 + math.cos(math.pi * step / total_steps))
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
    def __init__(self, optimizer, *, total_steps, min_lr=1e-5, last_epoch=-1, **kwargs):
        self.min_lr = min_lr

        super().__init__(optimizer=optimizer, total_steps=total_steps, last_epoch=last_epoch, **kwargs)

    def _get_lr(self, step):
        new_lrs = [
            _square_annealing(
                initial_lr=initial_lr,
                step=step - self.warmup_steps,
                total_steps=self.total_steps - self.warmup_steps,
                min_lr=self.min_lr,
            )
            for initial_lr in self.base_lrs
        ]
        return new_lrs


class SquareRootAnnealing(WarmupPolicy):
    def __init__(self, optimizer, *, total_steps, min_lr=0, last_epoch=-1, **kwargs):
        self.min_lr = min_lr

        super().__init__(optimizer=optimizer, total_steps=total_steps, last_epoch=last_epoch, **kwargs)

    def _get_lr(self, step):
        new_lrs = [
            _squareroot_annealing(initial_lr=initial_lr, step=step, total_steps=self.total_steps, min_lr=self.min_lr,)
            for initial_lr in self.base_lrs
        ]
        return new_lrs


class CosineAnnealing(WarmupPolicy):
    def __init__(self, optimizer, *, total_steps, min_lr=0, last_epoch=-1, **kwargs):
        self.min_lr = min_lr

        super().__init__(optimizer=optimizer, total_steps=total_steps, last_epoch=last_epoch, **kwargs)

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
                total_steps=self.total_steps - self.warmup_steps,
                min_lr=self.min_lr,
            )
            for initial_lr in self.base_lrs
        ]
        return new_lrs


class WarmupAnnealing(WarmupPolicy):
    def __init__(self, optimizer, *, total_steps, last_epoch=-1, **kwargs):
        super().__init__(optimizer=optimizer, total_steps=total_steps, last_epoch=last_epoch, **kwargs)

    def _get_lr(self, step):
        progress = float(step / self.total_steps)
        warmup_ratio = float(self.warmup_steps / self.total_steps)

        mult = max((progress - 1.0) / (warmup_ratio - 1.0), 0.0)
        out_lr = [initial_lr * mult for initial_lr in self.base_lrs]

        return out_lr


class InverseSquareRootAnnealing(WarmupPolicy):
    def __init__(self, optimizer, *, total_steps, last_epoch=-1, **kwargs):
        super().__init__(optimizer=optimizer, total_steps=total_steps, **kwargs, last_epoch=last_epoch)

    def _get_lr(self, step):
        denom = ((step + 1) / (self.warmup_steps + 1)) ** 0.5
        out_lr = [initial_lr / denom for initial_lr in self.base_lrs]
        return out_lr


class PolynomialDecayAnnealing(WarmupPolicy):
    def __init__(self, optimizer, *, total_steps, min_lr=0.0, power=1.0, cycle=False, last_epoch=-1, **kwargs):
        self.min_lr = min_lr
        self.power = power
        self.cycle = cycle

        super().__init__(optimizer=optimizer, total_steps=total_steps, last_epoch=last_epoch, **kwargs)

    def _get_lr(self, step):
        new_lrs = [
            _poly_decay(
                initial_lr,
                step=step - self.warmup_steps,
                decay_steps=self.total_steps - self.warmup_steps,
                power=self.power,
                min_lr=self.min_lr,
                cycle=self.cycle,
            )
            for initial_lr in self.base_lrs
        ]
        return new_lrs


class PolynomialHoldDecayAnnealing(WarmupHoldPolicy):
    def __init__(self, optimizer, *, total_steps, min_lr=0.0, power=1.0, cycle=False, last_epoch=-1, **kwargs):
        self.min_lr = min_lr
        self.power = power
        self.cycle = cycle

        super().__init__(optimizer=optimizer, total_steps=total_steps, last_epoch=last_epoch, **kwargs)

    def _get_lr(self, step):
        new_lrs = [
            _poly_decay(
                initial_lr,
                step=step - self.hold_steps,
                decay_steps=self.total_steps - max(self.warmup_steps, self.hold_steps),
                power=self.power,
                min_lr=self.min_lr,
                cycle=self.cycle,
            )
            for initial_lr in self.base_lrs
        ]
        return new_lrs
