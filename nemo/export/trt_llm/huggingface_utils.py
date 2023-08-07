# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import copy
from typing import List, Tuple

import numpy as np
import torch.nn as nn
from tensorrt_llm._utils import str_dtype_to_trt
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from .decoder import build_decoder_layer_config
from .model_config import (
    LINEAR_COLUMN,
    EmbeddingConfig,
    LayernormConfig,
    LinearConfig,
    ModelConfig,
)
from .tensor_utils import split, torch_to_numpy_with_dtype


def _arch_to_decoder_type(arch: str):
    arch_to_type = {
        "GPT2LMHeadModel": "gpt2",
        "GPTJForCausalLM": "gptj",
        "LlamaForCausalLM": "llama",
    }
    return arch_to_type.get(arch, "")


def _check_model_compatibility(model: nn.Module) -> Tuple[bool, bool]:
    """Returns whether the model is supported with the torch_to_tensorrt_llm API

    And if positional embedding layer exists.

    We assumes the model to be assembled with one or two embedding layers,
    a ModuleList of transformer decoders,
    and a final layernorm.
    Otherwise it will not be supported.
    """
    num_embeddings = 0
    num_module_list = 0
    num_layer_norm = 0
    for module in model.children():
        if type(module) == nn.Embedding:
            num_embeddings += 1
        elif type(module) == nn.ModuleList:
            num_module_list += 1
        elif type(module) in [nn.LayerNorm, LlamaRMSNorm]:
            num_layer_norm += 1

    return (
        1 <= num_embeddings
        and num_embeddings <= 2
        and num_module_list == 1
        and num_layer_norm == 1,
        num_embeddings > 1,
    )


def _get_transformer_model(model: nn.Module) -> nn.Module:
    """Returns the root module of the transformer model."""
    if hasattr(model, "transformer"):
        # This is a LMHead model
        return model.transformer
    elif hasattr(model, "model"):
        # LLAMA
        return model.model
    return model


def torch_to_model_config(
    model: nn.Module,
    gpus: int = 1,
) -> List[ModelConfig]:
    """The API to convert a torch or huggingface model to the ModelConfig format.

    The model has to be an LLM that we support for a successful conversion.
    (See examples/deploy/llm/README.md.)
    gpus: the number of inference gpus for multi gpu inferencing.

    Returns:
        The list of converted ModelConfig, one for each gpu.
    """
    transformer = _get_transformer_model(model)

    compatible, has_positional_embedding = _check_model_compatibility(transformer)
    assert compatible, f"model {transformer} not supported"

    assert (
        model.config.architectures and len(model.config.architectures) >= 1
    ), f"Huggingface model config {model.config} does not have architectures"

    model_config_template = ModelConfig()
    model_config_template.dtype = "float16"
    dtype = str_dtype_to_trt(model_config_template.dtype)

    model_config_template.tensor_parallel = gpus

    for name, module in transformer.named_children():
        if type(module) == nn.Embedding:
            if name != "wpe":
                model_config_template.vocab_embedding = EmbeddingConfig.from_nn_module(
                    module, dtype=dtype
                )
            else:
                assert has_positional_embedding
                model_config_template.positional_embedding = EmbeddingConfig.from_nn_module(
                    module, dtype=dtype
                )
        if type(module) in [nn.LayerNorm, LlamaRMSNorm]:
            model_config_template.final_layernorm = LayernormConfig.from_nn_module(
                module, dtype=dtype
            )

    model_configs = []
    for i in range(gpus):
        model_configs.append(copy.deepcopy(model_config_template))
        model_configs[i].rank = i

    decoder_type = _arch_to_decoder_type(model.config.architectures[0])
    for name, module in transformer.named_children():
        if type(module) == nn.ModuleList:
            for layer in module:
                for i in range(gpus):
                    model_configs[i].layers.append(
                        build_decoder_layer_config(
                            layer, decoder_type, rank=i, tensor_parallel=gpus, dtype=dtype
                        )
                    )

    if hasattr(model, "lm_head"):
        lm_head_weight = torch_to_numpy_with_dtype(model.lm_head.weight, dtype=dtype)
    else:
        # We use wte weights if not provided.
        lm_head_weight = model_configs[0].vocab_embedding.weight

    if model_configs[0].vocab_size_padded != model_configs[0].vocab_size:
        pad_width = model_configs[0].vocab_size_padded - model_configs[0].vocab_size
        lm_head_weight = np.pad(
            lm_head_weight, ((0, pad_width), (0, 0)), "constant", constant_values=0
        )

    for i in range(gpus):
        model_configs[i].lm_head = LinearConfig(linear_type=LINEAR_COLUMN)
        model_configs[i].lm_head.weight = np.ascontiguousarray(
            split(lm_head_weight, model_configs[i].tensor_parallel, model_configs[i].rank)
        )

    return model_configs
