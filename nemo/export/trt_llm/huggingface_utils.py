# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import os
import shutil
from typing import Tuple

import torch.multiprocessing as mp
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from .tensorrt_llm_model import LMHeadModelBuilder


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


def _torch_to_tensorrt_llm_impl(
    rank: int,
    model: nn.Module,
    engine_dir: str,
    gpus: int = 1,
    max_input_len=200,
    max_output_len=200,
    max_batch_size=1,
    max_beam_width=1,
    parallel_build=False,
):
    """The implmenetation of torch_to_tensorrt_llm for a single rank."""
    if hasattr(model, "transformer"):
        # This is a LMHead model
        transformer = model.transformer
    elif hasattr(model, "model"):
        # LLAMA
        transformer = model.model
    else:
        transformer = model

    compatible, has_positional_embedding = _check_model_compatibility(transformer)
    assert compatible, f"model {transformer} not supported"

    builder = LMHeadModelBuilder(rank=rank, tensor_parallel=gpus)
    for name, module in transformer.named_children():
        if type(module) == nn.Embedding:
            if name != "wpe":
                builder.add_vocab_embedding(module)
            else:
                assert has_positional_embedding
                builder.add_positional_embedding(module)
        if type(module) == nn.ModuleList:
            builder.add_decoder_layers(module)
        if type(module) in [nn.LayerNorm, LlamaRMSNorm]:
            builder.add_final_layernorm(module)

    if hasattr(model, "lm_head"):
        builder.finalize(model.lm_head)
    else:
        builder.finalize()

    builder.build(
        output_dir=engine_dir,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        max_batch_size=max_batch_size,
        max_beam_width=max_beam_width,
        parallel_build=parallel_build,
    )


def torch_to_tensorrt_llm(
    model: nn.Module,
    engine_dir: str,
    gpus: int = 1,
    max_input_len=200,
    max_output_len=200,
    max_batch_size=1,
    max_beam_width=1,
    parallel_build=False,
):
    """The API to convert a torch or huggingface model to tensorrt_llm.

    The model has to be an LLM that we support for a successful conversion.
    (See examples/deploy/llm/README.md.)
    gpus: the number of inference gpus for multi gpu inferencing.
    parallel_build: whether to build the multi gpu inference engine.
      Parallel build reduces the build time but increase the system memory load.
    """
    if os.path.exists(engine_dir):
        shutil.rmtree(engine_dir)

    if gpus == 1:
        _torch_to_tensorrt_llm_impl(
            0,
            model,
            engine_dir,
            gpus=1,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
        )
    elif parallel_build:
        mp.spawn(
            _torch_to_tensorrt_llm_impl,
            nprocs=gpus,
            args=(
                model,
                engine_dir,
                gpus,
                max_input_len,
                max_output_len,
                max_batch_size,
                max_beam_width,
                parallel_build,
            ),
        )
    else:
        for rank in range(gpus):
            _torch_to_tensorrt_llm_impl(
                rank,
                model,
                engine_dir,
                gpus=gpus,
                max_input_len=max_input_len,
                max_output_len=max_output_len,
                max_batch_size=max_batch_size,
                max_beam_width=max_beam_width,
                parallel_build=parallel_build,
            )
