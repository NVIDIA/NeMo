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

from typing import Callable, Optional

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    TorchFullyShardedDataParallel,
)
from megatron.core.enums import ModelType
from megatron.core.fp8_utils import is_float8tensor
from megatron.core.transformer.module import Float16Module

from nemo.collections.llm.gpt.model.base import GPTConfig
from nemo.collections.llm.t5.model.t5 import T5Config


def get_model_from_config(
    model_config: GPTConfig | T5Config,
    ddp_config: DistributedDataParallelConfig,
    overlap_param_gather_with_optimizer_step: bool = False,
    use_torch_fsdp2: bool = False,
    wrap_with_ddp: bool = True,
    data_parallel_random_init: bool = True,
    model_post_init_fns: Optional[list[Callable]] = None,
):
    # This method should only be called after `init_distributed()`.
    # model_provider_func is equivalent to llm.gpt.GPTConfig.configure_model()
    # model_type is inferred from the model_config class

    model_type = _get_model_type(model_config)
    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
    ):
        assert (
            model_type != ModelType.encoder_and_decoder
        ), "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(parallel_state.get_virtual_pipeline_model_parallel_world_size()):
            parallel_state.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = parallel_state.is_pipeline_first_stage()
            post_process = parallel_state.is_pipeline_last_stage()
            this_model = model_config.configure_model(
                tokenizer=None,
                pre_process=pre_process,
                post_process=post_process,
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = parallel_state.is_pipeline_first_stage()
        post_process = parallel_state.is_pipeline_last_stage()
        if model_type == ModelType.encoder_and_decoder:
            assert isinstance(model_config, T5Config)
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                rank = parallel_state.get_pipeline_model_parallel_rank()
                first_decoder_rank = parallel_state.get_pipeline_model_parallel_decoder_start()
                world_size = parallel_state.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == first_decoder_rank
                post_process = (rank == (first_decoder_rank - 1)) or (rank == (world_size - 1))
            model = model_config.configure_model(
                tokenizer=None,
            )
        else:
            model = model_config.configure_model(
                tokenizer=None,
                pre_process=pre_process,
                post_process=post_process,
            )
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if parallel_state.get_data_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) model parallel rank ({}, {}): {}".format(
                parallel_state.get_tensor_model_parallel_rank(),
                parallel_state.get_pipeline_model_parallel_rank(),
                sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model]),
            ),
            flush=True,
        )

    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if model_config.fp16 or model_config.bf16:
        model = [Float16Module(model_config, model_module) for model_module in model]

    if model_post_init_fns:
        for model_module in model:
            for post_init_fn in model_post_init_fns:
                post_init_fn(model_module)

    # The model_module.bfloat16()/model_module.half() above will call the inplace copy of TE's
    # Float8Tensor, which will write an unwanted value (amax calculated from the current fp8
    # param) to its amax_history. The following logic will correct the amax_history back.
    for model_module in model:
        for param in model_module.parameters():
            if is_float8tensor(param) and param._fp8_meta is not None:
                fp8_meta = param._fp8_meta["scaling_fwd"]
                fp8_meta_index = param._fp8_meta_index
                if hasattr(param, "get_high_precision_init_val"):
                    fp8_meta.amax_history[0][fp8_meta_index].copy_(param.get_high_precision_init_val().abs().max())
                else:
                    fp8_meta.amax_history[0][fp8_meta_index] = 0

    if wrap_with_ddp:
        if use_torch_fsdp2:
            DP = TorchFullyShardedDataParallel
        else:
            DP = DistributedDataParallel

        model = [
            DP(
                config=model_config,
                ddp_config=ddp_config,
                module=model_chunk,
                # Turn off bucketing for model_chunk 2 onwards, since communication for these
                # model chunks is overlapped with compute anyway.
                disable_bucketing=(model_chunk_idx > 0) or overlap_param_gather_with_optimizer_step,
            )
            for (model_chunk_idx, model_chunk) in enumerate(model)
        ]

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()
    return model


def _get_model_type(model_config: GPTConfig | T5Config) -> ModelType:
    return ModelType.encoder_and_decoder if isinstance(model_config, T5Config) else ModelType.encoder_or_decoder
