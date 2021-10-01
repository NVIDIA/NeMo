"""
This module support timing of code blocks.
"""
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import time

import numpy as np
import torch

__all__ = ["NamedTimer"]


class NamedTimer(object):
    """
    A timer class that supports multiple named timers.
    A named timer can be used multiple times, in which case the average
    dt will be returned.
    A named timer cannot be started if it is already currently running.
    Use case: measuring execution of multiple code blocks.
    """

    _REDUCTION_TYPE = ["mean", "sum", "min", "max", "none"]

    def __init__(self, reduction="mean", sync_cuda=False):
        """
        Args:
            reduction (str): reduction over multiple timings of the same timer
                             (none - returns the list instead of a scalar)
        """
        if reduction not in self._REDUCTION_TYPE:
            raise ValueError(f"Unknown reduction={reduction} please use one of {self._REDUCTION_TYPE}")

        self._reduction = reduction
        self._sync_cuda = sync_cuda

        self.reset()

    def __getitem__(self, k):
        return self.get(k)

    @property
    def _reduction_fn(self):
        if self._reduction == "none":
            fn = lambda x: x
        else:
            fn = getattr(np, self._reduction)

        return fn

    def reset(self, name=None):
        """
        Resents all / specific timer

        Args:
            name (str): timer name to reset (if None all timers are reset)
        """
        if name is None:
            self.timers = {}
        else:
            self.timers[name] = {}

    def start(self, name=""):
        """
        Starts measuring a named timer.

        Args:
            name (str): timer name to start
        """
        timer_data = self.timers.get(name, {})

        if "start" in timer_data:
            raise RuntimeError(f"Cannot start timer = '{name}' since it is already active")

        # synchronize pytorch cuda execution if supported
        if self._sync_cuda and torch.cuda.is_initialized():
            torch.cuda.synchronize()

        timer_data["start"] = time.time()

        self.timers[name] = timer_data

    def stop(self, name=""):
        """
        Stops measuring a named timer.

        Args:
            name (str): timer name to stop
        """
        timer_data = self.timers.get(name, None)
        if (timer_data is None) or ("start" not in timer_data):
            raise RuntimeError(f"Cannot end timer = '{name}' since it is not active")

        # synchronize pytorch cuda execution if supported
        if self._sync_cuda and torch.cuda.is_initialized():
            torch.cuda.synchronize()

        # compute dt and make timer inactive
        dt = time.time() - timer_data.pop("start")

        # store dt
        timer_data["dt"] = timer_data.get("dt", []) + [dt]

        self.timers[name] = timer_data

    def active_timers(self):
        """
        Return list of all active named timers
        """
        return [k for k, v in self.timers.items() if ("start" in v)]

    def get(self, name=""):
        """
        Returns the value of a named timer

        Args:
            name (str): timer name to return
        """
        dt_list = self.timers[name].get("dt", [])

        return self._reduction_fn(dt_list)

    def export(self):
        """
        Exports a dictionary with average/all dt per named timer
        """
        fn = self._reduction_fn

        data = {k: fn(v["dt"]) for k, v in self.timers.items() if ("dt" in v)}

        return data
