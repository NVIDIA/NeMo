# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from pathlib import Path
from typing import List

import numpy as np
import tensorrt as trt
import tensorrt_llm
import torch
from tensorrt_llm._utils import np_bfloat16


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


def get_tensor_from_file(dir_path: Path, name: str, dtype, shape: List = None) -> np.array:
    """Loads tensor saved in a file to a numpy array."""
    if dtype == trt.float16:
        np_dtype = np.float16
    elif dtype == trt.float32:
        np_dtype = np.float32
    elif dtype == trt.bfloat16:
        np_dtype = np_bfloat16
    else:
        assert False, f"{dtype} not supported"

    p = dir_path / f"model.{name}.bin"
    if not Path(p).exists():
        return None

    t = np.fromfile(p, dtype=np_dtype)
    if shape is not None:
        t = t.reshape(shape)
    return t
