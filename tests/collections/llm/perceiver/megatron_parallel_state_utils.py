"""Minimal utilities for managing Megatron parallel state in tests."""

import os
from contextlib import contextmanager

import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel import random as tp_random


def clean_up_distributed_and_parallel_states():
    """Clean up parallel states, torch.distributed and torch cuda cache."""
    parallel_state.destroy_model_parallel()  # destroy parallel state before distributed
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


@contextmanager
def distributed_model_parallel_state(
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
    backend: str = "nccl",
    **initialize_model_parallel_kwargs,
):
    """Context manager for torch distributed and parallel state testing.

    Args:
        seed (int): random seed for tensor parallel RNG. Defaults to 42.
        rank (int): global rank of current device. Defaults to 0.
        world_size (int): world size/number of devices. Defaults to 1.
        backend (str): backend for torch.distributed. Defaults to 'nccl'.
        **initialize_model_parallel_kwargs: kwargs for initialize_model_parallel.
    """
    initial_states = None
    try:
        clean_up_distributed_and_parallel_states()

        # Set up distributed environment
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        os.environ["RANK"] = str(rank)

        # Initialize distributed and parallel state
        torch.distributed.init_process_group(backend=backend, world_size=world_size)
        parallel_state.initialize_model_parallel(**initialize_model_parallel_kwargs)

        # Set up tensor parallel RNG
        if tp_random.get_cuda_rng_tracker().is_initialized():
            initial_states = tp_random.get_cuda_rng_tracker().get_states()
        if seed is not None:
            tp_random.model_parallel_cuda_manual_seed(seed)

        yield

    finally:
        # Restore RNG state
        if initial_states is not None:
            tp_random.get_cuda_rng_tracker().set_states(initial_states)
        else:
            tp_random.get_cuda_rng_tracker().reset()

        clean_up_distributed_and_parallel_states() 