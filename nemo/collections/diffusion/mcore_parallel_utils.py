# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""
Megatron Model Parallel Initialization
"""

import os
import megatron.core.parallel_state as ps
import torch

class Utils:

    world_size = torch.cuda.device_count()
    rank = 0

    @staticmethod
    def initialize_distributed(tensor_model_parallel_size=1, 
                              pipeline_model_parallel_size=1,
                              context_parallel_size=1):
        ps.destroy_model_parallel()

        # Torch setup for distributed training
        rank = int(os.environ['LOCAL_RANK'])
        world_size = torch.cuda.device_count()
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(world_size=world_size, rank=rank)

        # Megatron core distributed training initialization
        ps.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size=context_parallel_size)
        
    @staticmethod
    def set_world_size(world_size=None, rank=None):
        Utils.world_size = torch.cuda.device_count() if world_size is None else world_size
        if torch.distributed.is_initialized() and Utils.world_size != torch.distributed.get_world_size():
            torch.distributed.destroy_process_group()

        if rank is None:
            Utils.rank = 0
            if Utils.rank >= Utils.world_size:
                Utils.rank = -1
        else:
            Utils.rank = rank

    @staticmethod
    def destroy_model_parallel():
        ps.destroy_model_parallel()
        torch.distributed.barrier()

    @staticmethod
    def initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_split_rank=None,
        **kwargs,
    ):
        ps.destroy_model_parallel()
        Utils.initialize_distributed()
        ps.initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank,
            **kwargs,
        )