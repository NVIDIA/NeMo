# Copyright (c) 2019 NVIDIA Corporation
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

import inspect
import math
import sys
from abc import ABC, abstractmethod


class _LRPolicy(ABC):
    """Base class for defining learning rate policies"""

    @abstractmethod
    def __call__(self, initial_lr, step, epoch):
        """Calculate and return learning rate for current epoch and step.

        Args:
            initial_lr: Initial learning rate to be passed to optimizer
            step: Current step number
            epoch: Current epoch number

        Returns:
            Learning rate value as single float number

        """

        pass


class WarmupPolicy(_LRPolicy):
    """Adds warmup kwargs and warmup logic to lr policy.

    All arguments should be passed as kwargs for clarity,

    Args:
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps
        total_steps: Total number of steps while training or `None` for
            infinite training

    """

    def __init__(self, *, warmup_steps=None, warmup_ratio=None, total_steps=None):
        assert not (
            warmup_steps is not None and warmup_ratio is not None
        ), "Either use particular number of step or ratio"
        assert warmup_ratio is None or total_steps is not None, "If there is a ratio, there should be a total steps"

        super().__init__()

        self.total_steps = total_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * total_steps)
        else:
            self.warmup_steps = 0

    def __call__(self, initial_lr, step, epoch):
        if step < self.warmup_steps:
            return initial_lr * (step + 1) / (self.warmup_steps + 1)
        if step > self.total_steps:
            return 0
        return self._get_lr(initial_lr, step, epoch)

    def _get_lr(self, initial_lr, step, epoch):
        """Simple const lr policy"""
        return initial_lr


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
        self, *, warmup_steps=None, warmup_ratio=None, hold_steps=None, hold_ratio=None, total_steps=None, min_lr=0.0
    ):
        assert not (hold_steps is not None and hold_ratio is not None), "Either use particular number of step or ratio"
        assert hold_ratio is None or total_steps is not None, "If there is a ratio, there should be a total steps"
        super().__init__(warmup_steps=warmup_steps, warmup_ratio=warmup_ratio, total_steps=total_steps)

        self._min_lr = min_lr
        self._last_warmup_lr = 0.0

        if hold_steps is not None:
            self.hold_steps = hold_steps + self.warmup_steps
        elif hold_ratio is not None:
            self.hold_steps = int(hold_ratio * total_steps) + self.warmup_steps
        else:
            self.hold_steps = 0

    def __call__(self, initial_lr, step, epoch):
        # Warmup phase
        if step < self.warmup_steps:
            warmup_lr = initial_lr * (step + 1) / (self.warmup_steps + 1)
            return warmup_lr

        # Hold phase
        if (step >= self.warmup_steps) and (step < self.hold_steps):
            return initial_lr

        if step > self.total_steps:
            return self._min_lr
        return self._get_lr(initial_lr, step, epoch)


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
    def __init__(self, total_steps, min_lr=1e-5, **kwargs):
        super().__init__(total_steps=total_steps, **kwargs)
        self.min_lr = min_lr

    def _get_lr(self, initial_lr, step, epoch):
        return _square_annealing(
            initial_lr=initial_lr,
            step=step - self.warmup_steps,
            total_steps=self.total_steps - self.warmup_steps,
            min_lr=self.min_lr,
        )


class SquareRootAnnealing(WarmupPolicy):
    def __init__(self, total_steps, min_lr=0, **kwargs):
        super().__init__(total_steps=total_steps, **kwargs)
        self.min_lr = min_lr

    def _get_lr(self, initial_lr, step, epoch):
        return _squareroot_annealing(
            initial_lr=initial_lr, step=step, total_steps=self.total_steps, min_lr=self.min_lr,
        )


class CosineAnnealing(WarmupPolicy):
    def __init__(self, total_steps, min_lr=0, **kwargs):
        self.min_lr = min_lr
        super().__init__(total_steps=total_steps, **kwargs)

    def _get_lr(self, initial_lr, step, epoch):
        if initial_lr < self.min_lr:
            raise ValueError(
                f"{self} received an initial learning rate that " f"was lower than the minimum learning rate."
            )
        return _cosine_annealing(
            initial_lr=initial_lr,
            step=step - self.warmup_steps,
            total_steps=self.total_steps - self.warmup_steps,
            min_lr=self.min_lr,
        )


class WarmupAnnealing(WarmupPolicy):
    def __init__(self, total_steps, **kwargs):
        super().__init__(total_steps=total_steps, **kwargs)

    def _get_lr(self, initial_lr, step, epoch):
        progress = float(step / self.total_steps)
        warmup_ratio = float(self.warmup_steps / self.total_steps)

        mult = max((progress - 1.0) / (warmup_ratio - 1.0), 0.0)
        out_lr = initial_lr * mult

        return out_lr


class InverseSquareRootAnnealing(WarmupPolicy):
    def __init__(self, total_steps, **kwargs):
        super().__init__(total_steps=total_steps, **kwargs)

    def _get_lr(self, initial_lr, step, epoch):
        denom = ((step + 1) / (self.warmup_steps + 1)) ** 0.5
        out_lr = initial_lr / denom
        return out_lr


class PolynomialDecayAnnealing(WarmupPolicy):
    def __init__(self, total_steps, min_lr=0.0, power=1.0, cycle=False, **kwargs):
        super().__init__(total_steps=total_steps, **kwargs)
        self.min_lr = min_lr
        self.power = power
        self.cycle = cycle

    def _get_lr(self, initial_lr, step, epoch):
        return _poly_decay(
            initial_lr,
            step=step - self.warmup_steps,
            decay_steps=self.total_steps - self.warmup_steps,
            power=self.power,
            min_lr=self.min_lr,
            cycle=self.cycle,
        )


class PolynomialHoldDecayAnnealing(WarmupHoldPolicy):
    def __init__(self, total_steps, min_lr=0.0, power=1.0, cycle=False, **kwargs):
        super().__init__(total_steps=total_steps, **kwargs)
        self.min_lr = min_lr
        self.power = power
        self.cycle = cycle

    def _get_lr(self, initial_lr, step, epoch):
        return _poly_decay(
            initial_lr,
            step=step - self.hold_steps,
            decay_steps=self.total_steps - max(self.warmup_steps, self.hold_steps),
            power=self.power,
            min_lr=self.min_lr,
            cycle=self.cycle,
        )


def get_all_lr_classes():
    """ Get all LR classes defined within this module
    """
    lr_classes = {}
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) and name != 'ABC':
            lr_classes[name] = obj
    return lr_classes


def get_lr_policy(lr_policy, **kwargs):
    lr_classes = get_all_lr_classes()
    if lr_policy not in lr_classes:
        raise ValueError(
            f'{lr_policy} is not a supported lr policy. ' f'Supported lr policies are {lr_classes.keys()}.'
        )
    return lr_classes[lr_policy](**kwargs)
