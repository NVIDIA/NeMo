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


from torch.optim.lr_scheduler import _LRScheduler
from nemo.utils import logging
from nemo.core.config.schedulers import (
    WarmupAnnealingHoldSchedulerParams,
)

class LinearWarmupHoldDecayPolicy(_LRScheduler):
    """
    Has a linear warmup phase from 0 to the optimizer's learning rate, then a
    constant phase, followed by a linear decay to the `min_lr`, after which
    the learning rate is held constant.

    E.g.,
    lr
    |        _____________  <- optimizer.lr
    |       /             \
    |      /               \
    |     /                 \
    |    /                   \
    |   /                     \
    |  /                       \_______________ <- min_lr
    | /
    |/                          |<- max_steps
    |------------------------------------------- steps
         |<- warmup_steps |<- constant_steps

    Args:
        optimizer: Optimizer for the scheduler. In the warmup stage, the
            scheduler linearly increases the learning rate from 0 to the rate
            specified by the optimizer.
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps. Mutually exclusive
            with `warmup_steps`
        max_steps: Total number of steps while training or `None` for
            infinite training
        min_lr: Minimum lr to hold the learning rate after decay at.
        constant_steps: Number of steps to keep lr constant at.
        constant_ratio: Ratio of steps to keep lr constant. Mutually exclusive
            with `constant_steps`.
    """

    def __init__(
        self,
        optimizer,
        *,
        warmup_steps=None,
        warmup_ratio=None,
        constant_steps=None,
        constant_ratio=None,
        max_steps=None,
        min_lr=0.0,
        last_epoch=-1,
    ):
        assert not (
            warmup_steps is not None and warmup_ratio is not None
        ), "Either use particular number of step or ratio"
        assert not (
            constant_steps is not None and constant_ratio is not None
        ), "Either use constant_steps or constant_ratio"
        assert warmup_ratio is None or max_steps is not None, "If there is a "\
            "ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps

        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        if constant_steps is not None:
            self.constant_steps = constant_steps
        elif constant_ratio is not None:
            self.constant_steps = int(constant_ratio * max_steps)
        else:
            self.constant_steps = 0

        self.decay_steps = max_steps - (
            self.constant_steps + self.warmup_steps)

        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            logging.warning(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.", UserWarning
            )

        step = self.last_epoch

        # Warmup steps
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return self._get_warmup_lr(step)

        # Constant steps after warmup
        if self.warmup_steps > 0 and step > self.warmup_steps and step <=  \
                (self.warmup_steps + self.constant_steps):
            return self._get_constant_lr(step)

        # Min lr after max steps of updates
        if step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        # decay after constant phase
        return self._get_decay_lr(step)

    def _get_warmup_lr(self, step):
        lr_val = (step + 1) / (self.warmup_steps + 1)
        return [initial_lr * lr_val for initial_lr in self.base_lrs]

    def _get_constant_lr(self, step):
        return [lr_val for lr_val in self.base_lrs]

    def _get_decay_lr(self, step):
        lr_val = (step - (self.warmup_steps + self.constant_steps) + 1) / \
            (self.decay_steps + 1)
        return [initial_lr * (1 - lr_val) for initial_lr in self.base_lrs]


class LinearWarmupHoldDecayParams(WarmupAnnealingHoldSchedulerParams):
    pass
