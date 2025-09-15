# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional

import torch

from nemo.utils import logging


COMPUTE_DTYPE_MAP = {
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
    'float32': torch.float32,
}


def setup_device(device: str, device_id: int, compute_dtype: str) -> tuple[str, Optional[int], torch.dtype]:
    """
    Set up the compute device for the model.

    Args:
        device: Requested device type ('cuda' or 'cpu').
        device_id: Requested CUDA device ID.
        compute_dtype: Requested compute dtype.

    Returns:
        Tuple of (device_string, device_id, compute_dtype) for model initialization.
    """
    device = device.strip()
    device_id = int(device_id) if device_id is not None else 0

    if torch.cuda.is_available() and device == "cuda":
        if device_id >= torch.cuda.device_count():
            logging.warning(f"Device ID {device_id} is not available. Using GPU 0 instead.")
            device_id = 0

        compute_dtype = COMPUTE_DTYPE_MAP.get(compute_dtype, None)
        if compute_dtype is None:
            raise ValueError(
                f"Invalid compute dtype: {compute_dtype}. Must be one of {list(COMPUTE_DTYPE_MAP.keys())}"
            )

        device_str = f"cuda:{device_id}"
        return device_str, device_id, compute_dtype

    if device == "cuda":
        logging.warning(f"Device {device} is not available. Using CPU instead.")

    return "cpu", -1, torch.float32
