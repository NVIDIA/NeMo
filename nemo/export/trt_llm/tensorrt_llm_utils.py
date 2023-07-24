# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from pathlib import Path
from typing import List

import numpy as np
import tensorrt as trt
import torch
from tensorrt_llm.layers import Embedding, LayerNorm, RmsNorm
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm


def get_hidden_size(nn_module):
    """Returns the hidden size from the nn_module."""
    if type(nn_module) == nn.Embedding:
        return nn_module.embedding_dim
    if type(nn_module) == nn.LayerNorm:
        return nn_module.normalized_shape[0]
    if type(nn_module) == LlamaRMSNorm:
        return nn_module.weight.shape[0]


def get_hidden_act(act_func):
    """Returns the name of the hidden activation functon based on ACT2FN."""
    if isinstance(act_func, str):
        return act_func

    for name, func in ACT2FN.items():
        if isinstance(func, tuple):
            if isinstance(act_func, func[0]):
                return name
        elif isinstance(act_func, func):
            return name
    assert False, f"Cannot find name for {act_func}"


def torch_to_np(tensor, dtype):
    """Convert a torch tensor to numpy array with the dtype."""
    if dtype == trt.float16:
        torch_dtype = torch.float16
    elif dtype == trt.float32:
        torch_dtype = torch.float32
    else:
        assert False, f"{dtype} not supported"
    return tensor.to(torch_dtype).cpu().detach().numpy()


def build_embedding(torch_module: nn.Embedding, dtype):
    """Returns the tensorrt_llm embedding layer from the torch embedding."""
    trt_embedding = Embedding(torch_module.num_embeddings, torch_module.embedding_dim, dtype=dtype)
    trt_embedding.weight.value = torch_to_np(torch_module.weight, dtype)
    return trt_embedding


def build_layernorm(torch_module: nn.Module, dtype):
    """Returns the tensorrt_llm layernorm layer from the torch layernorm"""
    if type(torch_module) == nn.LayerNorm:
        trt_layernorm = LayerNorm(normalized_shape=get_hidden_size(torch_module), dtype=dtype)
        trt_layernorm.weight.value = torch_to_np(torch_module.weight, dtype)
        trt_layernorm.bias.value = torch_to_np(torch_module.bias, dtype)
    elif type(torch_module) == LlamaRMSNorm:
        trt_layernorm = RmsNorm(normalized_shape=get_hidden_size(torch_module), dtype=dtype)
        trt_layernorm.weight.value = torch_to_np(torch_module.weight, dtype)
    else:
        raise NotImplementedError(f"{torch_module} not supported")
    return trt_layernorm


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
    else:
        assert False, f"{dtype} not supported"

    p = dir_path / f"model.{name}.bin"
    assert Path(p).exists(), f"{p} does not exist"

    t = np.fromfile(p, dtype=np_dtype)
    if shape is not None:
        t = t.reshape(shape)
    return t
