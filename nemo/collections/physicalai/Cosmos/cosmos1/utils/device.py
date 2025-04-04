# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os

import pynvml


class Device:
    """A class to handle NVIDIA GPU device operations using NVML.

    This class provides an interface to access and manage NVIDIA GPU devices,
    including retrieving device information and CPU affinity settings.

    Attributes:
        _nvml_affinity_elements (int): Number of 64-bit elements needed to represent CPU affinity
    """

    _nvml_affinity_elements = math.ceil(os.cpu_count() / 64)  # type: ignore

    def __init__(self, device_idx: int):
        """Initialize a Device instance for a specific GPU.

        Args:
            device_idx (int): Index of the GPU device to manage

        Raises:
            NVMLError: If the device cannot be found or initialized
        """
        super().__init__()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

    def get_cpu_affinity(self) -> list[int]:
        """Get the CPU affinity mask for this GPU device.

        Retrieves the CPU affinity mask indicating which CPU cores are assigned
        to this GPU device. The affinity is returned as a list of CPU core indices.

        Returns:
            list[int]: List of CPU core indices that have affinity with this GPU

        Raises:
            NVMLError: If the CPU affinity information cannot be retrieved

        Example:
            >>> device = Device(0)
            >>> device.get_cpu_affinity()
            [0, 1, 2, 3]  # Shows this GPU has affinity with CPU cores 0-3
        """
        affinity_string = ""
        for j in pynvml.nvmlDeviceGetCpuAffinity(self.handle, Device._nvml_affinity_elements):
            # assume nvml returns list of 64 bit ints
            affinity_string = "{:064b}".format(j) + affinity_string
        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is in 0th element of list
        return [i for i, e in enumerate(affinity_list) if e != 0]
