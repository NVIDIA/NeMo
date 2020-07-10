# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional


@dataclass
class SchedulerParams:
    """
    Base configuration for all schedulers.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    last_epoch: int = -1


@dataclass
class WarmupSchedulerParams(SchedulerParams):
    """
    Base configuration for all schedulers.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    warmup_steps: Optional[float] = None
    warmup_ratio: Optional[float] = None


@dataclass
class WarmupHoldSchedulerParams(WarmupSchedulerParams):
    """
    Base configuration for all schedulers.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    hold_steps: Optional[float] = None
    hold_ratio: Optional[float] = None
    min_lr: float = 0.0


@dataclass
class SquareAnnealingParams(WarmupSchedulerParams):
    """
    Square Annealing parameter config
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    min_lr: float = 1e-5


@dataclass
class SquareRootAnnealingParams(WarmupSchedulerParams):
    """
    Square Root Annealing parameter config
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    min_lr: float = 0.0


@dataclass
class CosineAnnealingParams(WarmupSchedulerParams):
    """
    Cosine Annealing parameter config
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    min_lr: float = 0.0


@dataclass
class WarmupAnnealingParams(WarmupSchedulerParams):
    """
    Warmup Annealing parameter config
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """


@dataclass
class InverseSquareRootAnnealingParams(WarmupSchedulerParams):
    """
    Inverse Square Root Annealing parameter config
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """


@dataclass
class PolynomialDecayAnnealingParams(WarmupSchedulerParams):
    """
    Polynomial Decay Annealing parameter config
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    power: float = 1.0
    cycle: bool = False


@dataclass
class PolynomialHoldDecayAnnealingParams(WarmupSchedulerParams):
    """
    Polynomial Hold Decay Annealing parameter config
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).
    """

    power: float = 1.0
    cycle: bool = False


def get_scheduler_config(name: str, **kwargs: Optional[Dict[str, Any]]) -> SchedulerParams:
    """
    Convenience method to obtain a SchedulerParams class and partially instantiate it with optimizer kwargs.

    Args:
        name: Name of the SchedulerParams in the registry.
        kwargs: Optional kwargs of the optimizer used during instantiation.

    Returns:
        a partially instantiated SchedulerParams
    """
    if name not in AVAILABLE_SCHEDULER_PARAMS:
        raise ValueError(
            f"Cannot resolve scheduler parameters '{name}'. Available scheduler parameters are : "
            f"{AVAILABLE_SCHEDULER_PARAMS.keys()}"
        )

    scheduler_params = AVAILABLE_SCHEDULER_PARAMS[name]
    scheduler_params = partial(scheduler_params, **kwargs)
    return scheduler_params


AVAILABLE_SCHEDULER_PARAMS = {
    'SchedulerParams': SchedulerParams,
    'WarmupPolicyParams': WarmupSchedulerParams,
    'WarmupHoldPolicyParams': WarmupHoldSchedulerParams,
    'SquareAnnealingParams': SquareAnnealingParams,
    'SquareRootAnnealingParams': SquareRootAnnealingParams,
    'InverseSquareRootAnnealingParams': InverseSquareRootAnnealingParams,
    'CosineAnnealingParams': CosineAnnealingParams,
    'WarmupAnnealingParams': WarmupAnnealingParams,
    'PolynomialDecayAnnealingParams': PolynomialDecayAnnealingParams,
    'PolynomialHoldDecayAnnealingParams': PolynomialHoldDecayAnnealingParams,
}
