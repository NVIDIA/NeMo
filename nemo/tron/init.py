import datetime
import time
import warnings
from typing import Callable, Optional

import torch
import torch.distributed
from megatron.core import parallel_state, tensor_parallel
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.core.utils import get_te_version, is_te_min_version

from nemo.tron.config import ConfigContainer, RerunStateMachineConfig
from nemo.tron.utils import get_local_rank_preinit, get_rank_safe, get_world_size_safe


def initialize_megatron(
    # extra_args_provider=None,
    # args_defaults={},
    # ignore_unknown_args=False,
    cfg: ConfigContainer,
    allow_no_cuda: bool = False,
    skip_mpu_initialization: bool = False,
    get_embedding_ranks: Optional[Callable[[list[int], Optional[int]], list[int]]] = None,
    get_position_embedding_ranks: Optional[Callable[[list[int], Optional[int]], list[int]]] = None,
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
        cfg.data_config.rampup_batch_size,
        cfg.data_config.global_batch_size,
        cfg.data_config.micro_batch_size,
        cfg.data_parallel_size,
        cfg.data_config.decrease_batch_size_if_needed,
    )

    # init rerun global state
    _init_rerun_state(cfg.rerun_state_machine_config)

    # torch.distributed initialization
    _torch_dist_init(cfg, get_embedding_ranks, get_position_embedding_ranks, skip_mpu_initialization)


def _torch_dist_init(
    cfg: ConfigContainer,
    get_embedding_ranks: Optional[Callable[[list[int], Optional[int]], list[int]]],
    get_position_embedding_ranks: Optional[Callable[[list[int], Optional[int]], list[int]]],
    skip_mpu_initialization: bool,
):
    def finish_mpu_init():
        # Pytorch distributed.
        _initialize_distributed(cfg, get_embedding_ranks, get_position_embedding_ranks)

        # Random seeds for reproducibility.
        if get_rank_safe() == 0:
            print("> setting random seeds to {} ...".format(cfg.rng_config.seed))
        _set_random_seed(
            cfg.rng_config.seed,
            cfg.megatron_lm_config.data_parallel_random_init,
            cfg.rng_config.te_rng_tracker,
            cfg.rng_config.inference_rng_tracker,
        )

    if skip_mpu_initialization:
        return None

    if cfg.megatron_lm_config.lazy_mpu_init:
        cfg.megatron_lm_config.use_cpu_initialization = True  # TODO (maanug): move to config validation
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        parallel_state.set_tensor_model_parallel_world_size(cfg.model_config.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        parallel_state.set_tensor_model_parallel_rank(get_rank_safe())
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        _compile_dataset_helpers()

        if cfg.model_config.tp_comm_overlap:
            _initialize_tp_communicators(cfg)

        # No continuation function
        return None


def _initialize_tp_communicators(cfg: ConfigContainer):
    """initializing the communicators with user buffers for high-performance tensor-model-parallel
    communication overlap"""

    try:
        import transformer_engine
        import yaml
        from transformer_engine.pytorch import module as te_module

    except ImportError:
        raise RuntimeError(
            "Tensor Parallel Communication/GEMM Overlap optimization needs 'yaml' and 'transformer_engine' packages"
        )

    if cfg.megatron_lm_config.tp_comm_overlap_cfg is not None:
        with open(cfg.megatron_lm_config.tp_comm_overlap_cfg, "r") as stream:
            ub_cfgs = yaml.safe_load(stream)
    else:
        ub_cfgs = {}

    input_shape = [
        (cfg.model_config.seq_length * cfg.data_config.micro_batch_size) // cfg.model_config.context_parallel_size,
        cfg.model_config.hidden_size,
    ]

    if is_te_min_version("1.9.0"):
        # The process group with the target bootstrap backend is created in Transformer Engine.
        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=cfg.model_config.tensor_model_parallel_size,
            use_fp8=(cfg.model_config.fp8 is not None),
            ub_cfgs=ub_cfgs,
            bootstrap_backend=cfg.model_config.tp_comm_bootstrap_backend,
        )
    else:
        if cfg.model_config.tp_comm_bootstrap_backend != "mpi":
            warnings.warn(f"Transformer Engine v{get_te_version()} supports only MPI bootstrap backend.")
        # Create a MPI process group to help with TP communication overlap bootstrap.
        torch.distributed.new_group(backend="mpi")

        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=cfg.model_config.tensor_model_parallel_size,
            use_fp8=(cfg.model_config.fp8 is not None),
            ub_cfgs=ub_cfgs,
        )


def _initialize_distributed(
    cfg: ConfigContainer,
    get_embedding_ranks: Optional[Callable[[list[int], Optional[int]], list[int]]],
    get_position_embedding_ranks: Optional[Callable[[list[int], Optional[int]], list[int]]],
):
    """Initialize torch.distributed and core model parallel."""

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        if get_rank_safe() == 0:
            print(
                "torch distributed is already initialized, skipping initialization ...",
                flush=True,
            )

    else:
        if get_rank_safe() == 0:
            print("> initializing torch distributed ...", flush=True)

        # Manually set the device ids.
        if device_count > 0:
            torch.cuda.set_device(get_local_rank_preinit())

        # Call the init process
        init_process_group_kwargs = {
            "backend": cfg.megatron_lm_config.distributed_backend,
            "world_size": get_world_size_safe(),
            "rank": get_rank_safe(),
            "timeout": datetime.timedelta(minutes=cfg.megatron_lm_config.distributed_timeout_minutes),
        }

        torch.distributed.init_process_group(**init_process_group_kwargs)

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if parallel_state.model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            parallel_state.initialize_model_parallel(
                cfg.model_config.tensor_model_parallel_size,
                cfg.model_config.pipeline_model_parallel_size,
                cfg.model_config.virtual_pipeline_model_parallel_size,
                cfg.model_config.pipeline_model_parallel_split_rank,
                context_parallel_size=cfg.model_config.context_parallel_size,
                hierarchical_context_parallel_sizes=cfg.model_config.hierarchical_context_parallel_sizes,
                expert_model_parallel_size=cfg.model_config.expert_model_parallel_size,
                num_distributed_optimizer_instances=cfg.ddp_config.num_distributed_optimizer_instances,
                expert_tensor_parallel_size=cfg.model_config.expert_tensor_parallel_size,
                distributed_timeout_minutes=cfg.megatron_lm_config.distributed_timeout_minutes,
                nccl_communicator_config_path=cfg.megatron_lm_config.nccl_communicator_config_path,
                order="tp-cp-ep-dp-pp" if not cfg.megatron_lm_config.use_tp_pp_dp_mapping else "tp-pp-dp",
                encoder_tensor_model_parallel_size=cfg.megatron_lm_config.encoder_tensor_model_parallel_size,
                encoder_pipeline_model_parallel_size=cfg.megatron_lm_config.encoder_pipeline_model_parallel_size,
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


def _init_rerun_state(rerun_state_machine_config: RerunStateMachineConfig):
    from megatron.core.rerun_state_machine import (
        RerunDiagnostic,
        RerunErrorInjector,
        RerunMode,
        initialize_rerun_state_machine,
    )

    def state_save_func():
        return {"rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states()}

    def state_restore_func(state_dict):
        if state_dict["rng_tracker_states"]:
            tensor_parallel.get_cuda_rng_tracker().set_states(state_dict["rng_tracker_states"])

    initialize_rerun_state_machine(
        state_save_func=state_save_func,
        state_restore_func=state_restore_func,
        mode=RerunMode(rerun_state_machine_config.rerun_mode),
        error_injector=RerunErrorInjector(
            error_injection_rate=rerun_state_machine_config.error_injection_rate,
            error_injection_type=RerunDiagnostic(rerun_state_machine_config.error_injection_type),
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
            ">>> done with dataset index builder. Compilation time: {:.3f} seconds".format(time.time() - start_time),
            flush=True,
        )


def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
):
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
