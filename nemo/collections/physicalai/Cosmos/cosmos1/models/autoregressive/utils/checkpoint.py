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

from typing import Dict, Optional

import torch

# Substrings to ignore when processing state dicts
substrings_to_ignore = [
    "_extra_state",  # Extra states (BytesIO type) added by TransformerEngine for FP8 handling
]


def get_partial_state_dict(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    """
    Get a partial state dict with keys starting with the given prefix
    """
    return {k: v for k, v in state_dict.items() if k.startswith(prefix)}


def process_state_dict(
    state_dict: Dict[str, torch.Tensor],
    device: str = None,
    dtype: torch.dtype = None,
    prefix_to_remove: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    - Remove items with substring "_extra_state" in keys (TransformerEngine adds these for FP8)
    - Move tensors to specified device and dtype if provided

    Args:
        state_dict (Dict[str, torch.Tensor]): The state dict to process
        device (str, optional): The device to move tensors to. Defaults to None.
        dtype (torch.dtype, optional): The dtype to move tensors to. Defaults to None.
        prefix_to_remove (str, optional): The prefix to remove from the keys of the state dict. Defaults to None.

    Returns:
        Dict[str, torch.Tensor]: The processed state dict
    """
    new_state_dict = {}
    tensor_kwargs = {}
    if device is not None:
        tensor_kwargs["device"] = device
    if dtype is not None:
        tensor_kwargs["dtype"] = dtype

    for key, value in state_dict.items():
        # Check if any of the substrings to ignore are in the key
        skip = False
        for substr in substrings_to_ignore:
            if substr in key:
                skip = True
                break
        if skip:
            continue
        if len(tensor_kwargs) > 0:
            value = value.to(**tensor_kwargs)
        if prefix_to_remove is not None and key.startswith(prefix_to_remove):
            key = key[len(prefix_to_remove) :]
        new_state_dict[key] = value
    return new_state_dict
