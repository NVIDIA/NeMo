import os
import time

import torch
from megatron.core import parallel_state, tensor_parallel

from nemo.tron.config import FlatConfig


def get_rank_safe() -> int:
    # In megatron init, args.rank comes from the torchrun env var.
    # Once init has been done, args.rank is updated to value of torch get_rank()
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return int(os.getenv('RANK', '0'))


def get_world_size_safe() -> int:
    # In megatron init, args.world_size comes from the torchrun env var.
    # Once init has been done, args.world_size is updated to value of torch get_world_size()
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return int(os.getenv("WORLD_SIZE", '1'))


def get_local_rank_preinit() -> int:
    return int(os.getenv('LOCAL_RANK', '0'))


def initialize_megatron(
    # extra_args_provider=None,
    # args_defaults={},
    # ignore_unknown_args=False,
    cfg: FlatConfig,
    allow_no_cuda=False,
    skip_mpu_initialization=False,
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
):
    """Initialize megatron global vars, logging, and distributed state."""

    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), "Megatron requires CUDA."

    # Prep for checkpoint conversion.
    # if args.ckpt_convert_format is not None:
    #     assert args.ckpt_convert_save is not None
    #     assert args.load is not None
    #     args.exit_on_missing_checkpoint = True

    # TODO (maanug): determine if we want to support this behavior
    # if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
    #     assert args.load is not None, "--use-checkpoint-args requires --load argument"
    #     load_args_from_checkpoint(args)

    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables()  # TODO (maanug): implement

    # init rerun global state
    _init_rerun_state(cfg)

    # torch.distributed initialization
    _torch_dist_init(cfg, get_embedding_ranks, get_position_embedding_ranks, skip_mpu_initialization)


def _torch_dist_init(cfg: FlatConfig, get_embedding_ranks, get_position_embedding_ranks, skip_mpu_initialization):
    def finish_mpu_init():
        # Pytorch distributed.
        _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks)  # TODO (maanug): implement

        # Random seeds for reproducibility.
        if get_rank_safe() == 0:
            print("> setting random seeds to {} ...".format(cfg.seed))
        _set_random_seed(cfg.seed, cfg.data_parallel_random_init, cfg.te_rng_tracker, cfg.inference_rng_tracker)

    if skip_mpu_initialization:
        return None

    if cfg.lazy_mpu_init:
        cfg.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        parallel_state.set_tensor_model_parallel_world_size(cfg.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        parallel_state.set_tensor_model_parallel_rank(get_rank_safe())
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        _compile_dataset_helpers()

        if cfg.tp_comm_overlap:
            # TODO: Should this be activated with just decoder-tp-comm-overlap too?
            _initialize_tp_communicators()  # TODO (maanug): implement

        # No continuation function
        return None


def _init_rerun_state(cfg: FlatConfig):
    from megatron.core.rerun_state_machine import (
        RerunDiagnostic,
        RerunErrorInjector,
        RerunMode,
        initialize_rerun_state_machine,
    )

    def state_save_func():
        return {'rng_tracker_states': tensor_parallel.get_cuda_rng_tracker().get_states()}

    def state_restore_func(state_dict):
        if state_dict['rng_tracker_states']:
            tensor_parallel.get_cuda_rng_tracker().set_states(state_dict['rng_tracker_states'])

    initialize_rerun_state_machine(
        state_save_func=state_save_func,
        state_restore_func=state_restore_func,
        mode=RerunMode(cfg.rerun_mode),
        error_injector=RerunErrorInjector(
            error_injection_rate=cfg.error_injection_rate,
            error_injection_type=RerunDiagnostic(cfg.error_injection_type),
        ),
    )


def _compile_dataset_helpers():

    # =========================
    # Compile dataset C++ code.
    # =========================
    # TODO: move this to ninja
    if get_rank_safe() == 0:
        start_time = time.time()
        print("> compiling dataset index builder ...")
        from megatron.core.datasets.utils import compile_helpers

        compile_helpers()
        print(
            ">>> done with dataset index builder. Compilation time: {:.3f} "
            "seconds".format(time.time() - start_time),
            flush=True,
        )


def _set_random_seed(seed_, data_parallel_random_init=False, te_rng_tracker=False, inference_rng_tracker=False):
    """Set random seed for reproducability."""
    assert seed_ is not None and seed_ > 0, f"Seed ({seed_}) should be a positive integer."

    import random
    import numpy as np

    # Ensure that different pipeline MP stages get different seeds.
    seed = seed_ + (100 * parallel_state.get_pipeline_model_parallel_rank())
    # Ensure different data parallel ranks get different seeds
    if data_parallel_random_init:
        seed = seed + (10 * parallel_state.get_data_parallel_rank())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        tensor_parallel.model_parallel_cuda_manual_seed(seed, te_rng_tracker, inference_rng_tracker)
