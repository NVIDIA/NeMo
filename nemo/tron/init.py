import datetime
import os
import time

import torch
import torch.distributed
from megatron.core import parallel_state, tensor_parallel
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator

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

    init_num_microbatches_calculator(
        get_rank_safe(),
        cfg.rampup_batch_size,
        cfg.global_batch_size,
        cfg.micro_batch_size,
        cfg.data_parallel_size,
        cfg.decrease_batch_size_if_needed,
    )

    # init rerun global state
    _init_rerun_state(cfg)

    # torch.distributed initialization
    _torch_dist_init(cfg, get_embedding_ranks, get_position_embedding_ranks, skip_mpu_initialization)


def _torch_dist_init(cfg: FlatConfig, get_embedding_ranks, get_position_embedding_ranks, skip_mpu_initialization):
    def finish_mpu_init():
        # Pytorch distributed.
        _initialize_distributed(cfg, get_embedding_ranks, get_position_embedding_ranks)

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


def _initialize_distributed(cfg: FlatConfig, get_embedding_ranks, get_position_embedding_ranks):
    """Initialize torch.distributed and core model parallel."""

    # NOTE (maanug): After this function is called,
    # can use torch.distributed.get_rank() instead of args.rank
    # can use torch.distributed.get_world_size() instead of args.world_size

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        if get_rank_safe() == 0:
            print(
                "torch distributed is already initialized, " "skipping initialization ...",
                flush=True,
            )

    else:
        if get_rank_safe() == 0:
            print("> initializing torch distributed ...", flush=True)

        # Manually set the device ids.
        if device_count > 0:
            torch.cuda.set_device(get_local_rank_preinit())
            device_id = torch.device(f'cuda:{get_local_rank_preinit()}')
        else:
            device_id = None

        # Call the init process
        init_process_group_kwargs = {
            'backend': cfg.distributed_backend,
            'world_size': get_world_size_safe(),
            'rank': get_rank_safe(),
            'timeout': datetime.timedelta(minutes=cfg.distributed_timeout_minutes),
        }

        torch.distributed.init_process_group(**init_process_group_kwargs)

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if parallel_state.model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            parallel_state.initialize_model_parallel(
                cfg.tensor_model_parallel_size,
                cfg.pipeline_model_parallel_size,
                cfg.virtual_pipeline_model_parallel_size,
                cfg.pipeline_model_parallel_split_rank,
                context_parallel_size=cfg.context_parallel_size,
                hierarchical_context_parallel_sizes=cfg.hierarchical_context_parallel_sizes,
                expert_model_parallel_size=cfg.expert_model_parallel_size,
                num_distributed_optimizer_instances=cfg.num_distributed_optimizer_instances,
                expert_tensor_parallel_size=cfg.expert_tensor_parallel_size,
                distributed_timeout_minutes=cfg.distributed_timeout_minutes,
                nccl_communicator_config_path=cfg.nccl_communicator_config_path,
                order='tp-cp-ep-dp-pp' if not cfg.use_tp_pp_dp_mapping else 'tp-pp-dp',
                encoder_tensor_model_parallel_size=cfg.encoder_tensor_model_parallel_size,
                encoder_pipeline_model_parallel_size=cfg.encoder_pipeline_model_parallel_size,
                get_embedding_ranks=get_embedding_ranks,
                get_position_embedding_ranks=get_position_embedding_ranks,
            )
            if get_rank_safe() == 0:
                print(
                    f"> initialized tensor model parallel with size "
                    f"{parallel_state.get_tensor_model_parallel_world_size()}"
                )
                print(
                    f"> initialized pipeline model parallel with size "
                    f"{parallel_state.get_pipeline_model_parallel_world_size()}"
                )


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
