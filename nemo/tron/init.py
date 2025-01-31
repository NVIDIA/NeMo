import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from megatron.core import parallel_state, tensor_parallel


@dataclass
class DistInitConfig:
    tensor_model_parallel_size: int = 1

    tp_comm_overlap: bool = False
    """Enables the overlap of Tensor parallel communication and GEMM kernels."""

    lazy_mpu_init: Optional[bool] = None
    """If set to True, initialize_megatron() skips DDP initialization and 
    returns function to complete it instead. Also turns on
    --use-cpu-initialization flag. This is for external DDP manager. """

    seed: int = 1234
    """Random seed used for python, numpy, pytorch, and cuda."""

    data_parallel_random_init: bool = False
    """Enable random initialization of params across data parallel ranks"""

    te_rng_tracker: bool = False
    """Use the Transformer Engine version of the random number generator. 
    Required for CUDA graphs support."""

    inference_rng_tracker: bool = False
    """Use a random number generator configured for inference."""


@dataclass
class RerunStateMachineConfig:
    error_injection_rate: int = 0
    """Rate at which to inject unexpected results, e.g. 1000 means 
    once every 1000 result validations"""

    error_injection_type: Literal['correct_result', 'transient_error', 'persistent_error'] = 'transient_error'
    """Type of error to inject. """

    rerun_mode: Literal['disabled', 'validate_results', 'report_stats'] = 'disabled'
    """Use re-run engine to validate results (default) or to emit stats
    on variability of computations due to non-deterministic algorithms."""


def get_rank_preinit() -> int:
    return int(os.getenv('RANK', '0'))


def initialize_megatron(
    # extra_args_provider=None,
    # args_defaults={},
    # ignore_unknown_args=False,
    rerun_sm_cfg: RerunStateMachineConfig,
    dist_cfg: DistInitConfig,
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
    _init_rerun_state(rerun_sm_cfg)

    # torch.distributed initialization
    _torch_dist_init(dist_cfg, get_embedding_ranks, get_position_embedding_ranks, skip_mpu_initialization)


def _torch_dist_init(cfg: DistInitConfig, get_embedding_ranks, get_position_embedding_ranks, skip_mpu_initialization):
    def finish_mpu_init():
        # Pytorch distributed.
        _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks)  # TODO (maanug): implement

        if get_rank_preinit() == 0:
            print("> setting random seeds to {} ...".format(cfg.seed))
        _set_random_seed(
            cfg.seed, cfg.data_parallel_random_init, cfg.te_rng_tracker, cfg.inference_rng_tracker
        )  # TODO (maanug): implement

    if skip_mpu_initialization:
        return None

    if cfg.lazy_mpu_init:
        # TODO (maanug): determine where this is accessed downstream
        # args.use_cpu_initialization = True

        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        parallel_state.set_tensor_model_parallel_world_size(cfg.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        parallel_state.set_tensor_model_parallel_rank(get_rank_preinit())
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        _compile_dependencies()  # TODO (maanug): implement

        if cfg.tp_comm_overlap:
            # TODO: Should this be activated with just decoder-tp-comm-overlap too?
            _initialize_tp_communicators()  # TODO (maanug): implement

        # No continuation function
        return None


def _init_rerun_state(cfg: RerunStateMachineConfig):
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
