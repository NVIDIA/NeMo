# -*- coding: utf-8 -*-

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

__all__ = ["skip_in_data_parallel", "run_only_on_device"]

from functools import partial

from nemo.core.neural_factory import DeviceType


def skip_in_data_parallel(cls):
    """ Decorator that adds the skip_in_data_parallel_ property (set to True) to neural module. """

    def decorator():
        setattr(cls, "skip_in_data_parallel", True)
        return cls

    return decorator


def run_only_on_device(cls=None, device_type: "DeviceType" = DeviceType.CPU):
    """ Decorator that adds the run_only_on_device property to neural module.
    Args:
        device_type: Type device to be set.
    """

    if cls is None:
        return partial(run_only_on_device, device_type=device_type)

    def decorator():
        setattr(cls, "run_only_on_device", device_type)
        return cls

    return decorator
