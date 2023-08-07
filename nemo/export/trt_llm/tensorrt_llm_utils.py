# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.


import tensorrt as trt
from tensorrt_llm.layers import Embedding, LayerNorm, RmsNorm
from tensorrt_llm.module import Module

from .model_config import (
    LAYERNORM_DEFAULT,
    LAYERNORM_RMS,
    EmbeddingConfig,
    LayernormConfig,
)
from .tensor_utils import get_tensor_parallel_group


def build_embedding_from_config(
    config: EmbeddingConfig, dtype: trt.DataType, tensor_parallel: int = 1
):
    """Returns the tensorrt_llm embedding layer from the embedding config."""
    # If the config is empty, return an empty impl.
    if config is None:
        return None
    trt_embedding = Embedding(
        config.weight.shape[0],
        config.weight.shape[1],
        dtype=dtype,
        tp_size=tensor_parallel,
        tp_group=get_tensor_parallel_group(tensor_parallel),
    )
    trt_embedding.weight.value = config.weight
    return trt_embedding


def build_layernorm_from_config(config: LayernormConfig, dtype: trt.DataType):
    """Returns the tensorrt_llm layernorm layer from the torch layernorm"""
    # If the config is empty, return an empty impl.
    if config is None:
        return None

    if config.layernorm_type == LAYERNORM_DEFAULT:
        trt_layernorm = LayerNorm(normalized_shape=config.weight.shape[0], dtype=dtype)
        trt_layernorm.weight.value = config.weight
        trt_layernorm.bias.value = config.bias
    elif config.layernorm_type == LAYERNORM_RMS:
        trt_layernorm = RmsNorm(normalized_shape=config.weight.shape[0], dtype=dtype)
        trt_layernorm.weight.value = config.weight
    else:
        raise NotImplementedError(f"{config.layernorm_type} not supported")
    return trt_layernorm


def print_tensorrt_llm(name: str, tensorrt_llm_module: Module):
    """Prints the tensorrt llm structure including weights and related data for debugging purpose."""
    if hasattr(tensorrt_llm_module, "weight") and tensorrt_llm_module.weight:
        print(f"{name}.weight:\n{tensorrt_llm_module.weight._value}")
    if hasattr(tensorrt_llm_module, "bias") and tensorrt_llm_module.bias:
        print(f"{name}.bias:\n{tensorrt_llm_module.bias._value}")
    if (
        hasattr(tensorrt_llm_module, "activation_scaling_factor")
        and tensorrt_llm_module.activation_scaling_factor
    ):
        print(
            f"{name}.activation_scaling_factor:\n{tensorrt_llm_module.activation_scaling_factor._value}"
        )
    if (
        hasattr(tensorrt_llm_module, "weights_scaling_factor")
        and tensorrt_llm_module.weights_scaling_factor
    ):
        print(
            f"{name}.weights_scaling_factor:\n{tensorrt_llm_module.weights_scaling_factor._value}"
        )

    for k, v in tensorrt_llm_module.named_children():
        print_tensorrt_llm(f"{name}.{k}({v._get_name()})", v)
