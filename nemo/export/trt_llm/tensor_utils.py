# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utils for tensor conversions between tensorrt, torch and numpy."""

from typing import Dict

import numpy as np
import tensorrt as trt
import tensorrt_llm
import torch


def torch_to_numpy_with_dtype(tensor, dtype=trt.float16):
    """Converts a torch tensor to numpy array with the dtype."""
    if dtype == trt.float16:
        torch_dtype = torch.float16
    elif dtype == trt.float32:
        torch_dtype = torch.float32
    elif dtype == trt.bfloat16:
        torch_dtype = torch.bfloat16
    else:
        assert False, f"{dtype} not supported"
    return tensorrt_llm._utils.torch_to_numpy(tensor.detach().to(torch_dtype))


def trt_dtype_to_str(dtype: trt.DataType):
    """Converts a trt dtype to string."""
    str_map = {
        trt.float16: "float16",
        trt.bfloat16: "bfloat16",
        trt.float32: "float32",
    }

    return str_map[dtype]


def split(v, tp_size, idx, dim=0):
    """Splits the np tensor v on dim and return the idx's slice."""
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    else:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])


def get_tensor_parallel_group(tensor_parallel: int):
    """Returns the tensor_parallel_group config based on tensor_parallel."""
    return None if tensor_parallel == 1 else list(range(tensor_parallel))


def get_tensor_from_dict(weights_dict: Dict[str, np.ndarray], name: str) -> np.array:
    """Loads tensor from the weights_dict."""
    return weights_dict.get(f"model.{name}.bin", None)
