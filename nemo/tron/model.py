from typing import Callable

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    TorchFullyShardedDataParallel,
)
from megatron.core.enums import ModelType
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import is_float8tensor

from nemo.tron.config import FlatConfig


def apply_parallel_wrappers(model_provider_func: Callable, cfg: FlatConfig, wrap_with_ddp=True):
    # This method should only be called after `init_distributed()`.
    # model_provider_func can be something like the current llm.gpt.GPTConfig.configure_model()
    # model_type can be something we infer instead of an argument

    model_cfg = cfg.to_module_cfg(TransformerConfig)

    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
    ):
        assert (
            cfg.model_type != ModelType.encoder_and_decoder
        ), "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(parallel_state.get_virtual_pipeline_model_parallel_world_size()):
            parallel_state.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = parallel_state.is_pipeline_first_stage()
            post_process = parallel_state.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
            )
            this_model.model_type = cfg.model_type
            model.append(this_model)
    else:
        pre_process = parallel_state.is_pipeline_first_stage()
        post_process = parallel_state.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if cfg.model_type == ModelType.encoder_and_decoder:
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                rank = parallel_state.get_pipeline_model_parallel_rank()
                first_decoder_rank = parallel_state.get_pipeline_model_parallel_decoder_start()
                world_size = parallel_state.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == first_decoder_rank
                post_process = (rank == (first_decoder_rank - 1)) or (rank == (world_size - 1))
                add_encoder = parallel_state.is_inside_encoder(rank)
                add_decoder = parallel_state.is_inside_decoder(rank)
            model = model_provider_func(
                pre_process=pre_process, post_process=post_process, add_encoder=add_encoder, add_decoder=add_decoder
            )
        else:
            model = model_provider_func(pre_process=pre_process, post_process=post_process)
        model.model_type = cfg.model_type

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
            ' > number of parameters on (tensor, pipeline) '
            'model parallel rank ({}, {}): {}'.format(
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
    if model_cfg.fp16 or model_cfg.bf16:
        model = [Float16Module(model_cfg, model_module) for model_module in model]

    # The model_module.bfloat16()/model_module.half() above will call the inplace copy of TE's
    # Float8Tensor, which will write an unwanted value (amax calculated from the current fp8
    # param) to its amax_history. The following logic will correct the amax_history back.
    for model_module in model:
        for param in model_module.parameters():
            if is_float8tensor(param) and param._fp8_meta is not None:
                fp8_meta = param._fp8_meta['scaling_fwd']
                fp8_meta_index = param._fp8_meta_index
                if hasattr(param, 'get_high_precision_init_val'):
                    fp8_meta.amax_history[0][fp8_meta_index].copy_(param.get_high_precision_init_val().abs().max())
                else:
                    fp8_meta.amax_history[0][fp8_meta_index] = 0

    if wrap_with_ddp:
        if cfg.use_torch_fsdp2:
            DP = TorchFullyShardedDataParallel
        else:
            DP = DistributedDataParallel
        # TODO: continue ddp wrap

        ddp_cfg = cfg.to_module_cfg(DistributedDataParallelConfig)

        model = [
            DP(
                config=model_cfg,
                ddp_config=ddp_cfg,
                module=model_chunk,
                # Turn off bucketing for model_chunk 2 onwards, since communication for these
                # model chunks is overlapped with compute anyway.
                disable_bucketing=(model_chunk_idx > 0) or cfg.overlap_param_gather_with_optimizer_step,
            )
            for (model_chunk_idx, model_chunk) in enumerate(model)
        ]

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if cfg.data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()

    return model
