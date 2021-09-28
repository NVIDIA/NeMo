"""
This module support timing of code blocks.
"""

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

    def __init__(self, mean=True):
        """
        mean - default behaviour. If True mean dt is returned, else a list.
        """
        self.mean = mean
        self.timers = {}

    def start(self, name=""):
        """
        Starts measuring a named timer.
        """
        timer_data = self.timers.get(name, {})

        if "start" in timer_data:
            raise RuntimeError(f"Cannot start timer = '{name}' since it is already active")

        # synchronize pytorch cuda execution if supported
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()

        timer_data["start"] = time.time()

        self.timers[name] = timer_data

    def stop(self, name=""):
        """
        Stops measuring a named timer.
        """
        timer_data = self.timers.get(name, None)
        if (timer_data is None) or ("start" not in timer_data):
            raise RuntimeError(f"Cannot end timer = '{name}' since it is not active")

        # synchronize pytorch cuda execution if supported
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()

        # compute dt and make timer inactive
        dt = time.time()-timer_data.pop("start")

        # store dt
        timer_data["dt"] = timer_data.get("dt", []) + [dt]

        self.timers[name] = timer_data


    def active(self):
        """
        Return list of all active named timers
        """
        return [k for k, v in self.timers.items() if ("start" in v)]

    def get(self, name="", mean=None):
        """
        Returns the value of a named timer
        """
        if mean is None:
            mean = self.mean

        if mean:
            fn = np.mean
        else:
            fn = lambda x: x

        dt_list = self.timer[name].get("dt", [])

        return fn(dt_list)

    def export(self, mean=None):
        """
        Exports a dictionary with average/all dt per named timer

        mean - if True return the mean per name of all measures,
               else return a list per name.
        """
        if mean is None:
            mean = self.mean

        if mean:
            fn = np.mean
        else:
            fn = lambda x: x

        data = {k: fn(v["dt"]) for k, v in self.timers.items() if ("dt" in v)}

        return data