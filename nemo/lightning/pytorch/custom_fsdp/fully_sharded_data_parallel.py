# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import functools
import logging
from contextlib import contextmanager
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from megatron.core import parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.distributed.custom_fsdp.param_and_grad_buffer import (
    AllGatherPipeline,
    BucketingPolicy,
    GradReducePipeline,
    ParamAndGradBuffer,
    PrefetchOrder,
)
from megatron.core.distributed.data_parallel_base import _BaseDataParallel
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.fp8_utils import is_float8tensor
from megatron.core.process_groups_config import GradCommProcessGroups
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import is_submodule, log_single_rank

logger = logging.getLogger(__name__)


class TrainingState(Enum):
    """States of a FSDP parameter group, which are coupled with
    the sharding activity of parameters and gradients during training."""

    # From pre-forward before post-forward, where parameters should be unsharded
    FORWARD = auto()
    # Prior to backward computation, where parameters should be unsharded
    PRE_BACKWARD = auto()
    # After backward computation, where gradients should be re-sharded
    POST_BACKWARD = auto()
    # Before and after module forward computaton or before pre-backward and
    # after post-backward states, where no un/sharding activity happens
    IDLE = auto()


class FullyShardedDataParallel(_BaseDataParallel):
    """Fully Sharded Data Parallel training for MCore models.

    A distributed training wrapper that shards model parameters, gradients and optimizer
    states across data parallel workers. Integrates seamlessly with MCore's tensor
    and expert parallelism features.

    We supports following modes:
    - no_shard: Traditional data parallel training without parameter sharding.
    - optim: Shards optimizer states, this is conceptually close to "ZeRO-1", and
        main weights for mixed precision training, meanwhile the following `optim_grads`
        and `optim_grads_params` will also sharding main weights
        during mixed-precision training, omitted without detailed notation.
    - optim_grads: Shards gradients and optimizer states, this is conceptually close to "ZeRO-2".
    - optim_grads_params: Shards parameters, gradients and optimizer states, this
        is conceptually close to "ZeRO-3".

    Key Features:
    - Compatible with MCore's tensor, context and expert parallelism
    - Automatic mixed precision training (BF16/FP8)
    - Gradient accumulation and bucketing
    - Optimized activation recompute with shard-aware communication: When recomputing
        a whole Transformer layer, gather parameters once for both the recomputation
        and backward computation
    - Compatible with MCore's distributed checkpointing

    Args:
        config: Transformer config object.
        ddp_config: FullyShardedDataParallel config object.
        module: Underlying model.
        fsdp_unit_modules: List of modules that should be treated as FSDP Unit,
            i.e., the minimum releasable model unit. If not provided, defaults to
            [TransformerLayer, LanguageModelEmbedding] for GPT-like models. In
            addition to this, it affects the granularity of the communication
            parameter grouping and triggers aggregate collective communication
            in fp8 mixed precision training.
        disable_bucketing: If true, force assign all parameters to a single bucket. If false,
            use standard bucketing policy: assign parameters to smaller buckets and all-reduce
            per bucket.
        grad_comm_pgs: Optional GradCommProcessGroups object. If not provided, the default
            process groups from parallel_state will be used. If provided, module expects
            grad_comm_pgs to have dp_cp or dp (if cp=1) and
            expt_dp attributes(if using expert data parallelism).
    Examples:
        >>> model = GPTModel(config)
        >>> model = FullyShardedDataParallel(
        ...     config,
        ...     model,
        ...     ddp_config,
        ...     fsdp_unit_modules = [TransformerLayer, LanguageModelEmbedding],
        ... )
    """

    # TODO: add hybrid FSDP (shard model states in a partial DP domain)
    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        fsdp_unit_modules: Optional[List[torch.nn.Module]] = None,
        disable_bucketing: bool = False,
        device: Optional[torch.device] = None,
        grad_comm_pgs: Optional[GradCommProcessGroups] = None,
    ):
        super().__init__(config=config, module=module)
        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.module = module
        self.ddp_config = ddp_config
        log_single_rank(
            logger,
            logging.INFO,
            f'Setting up DistributedDataParallel with config {self.ddp_config}',
        )

        if grad_comm_pgs is None:
            self.dp_cp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=True, partial_data_parallel=False
            )
            self.expt_dp_group = parallel_state.get_expert_data_parallel_group()

        else:
            cp_size = getattr(config, 'context_parallel_size', 1)

            if hasattr(grad_comm_pgs, 'dp_cp'):
                self.dp_cp_group = grad_comm_pgs.dp_cp
            elif hasattr(grad_comm_pgs, 'dp') and cp_size == 1:
                self.dp_cp_group = grad_comm_pgs.dp
            else:
                raise ValueError(
                    "Required process group missing: 'dp_cp' (or 'dp' when context_parallel_size=1)"
                )

            have_expert_parameters = False
            for _, param in self.module.named_parameters():
                if not getattr(param, 'allreduce', True):
                    have_expert_parameters = True
                    break
            if have_expert_parameters:
                assert hasattr(
                    grad_comm_pgs, 'expt_dp'
                ), 'expert process group is required when using expert parameters'
                self.expt_dp_group = grad_comm_pgs.expt_dp
            else:
                self.expt_dp_group = None

        self.bucket_size = self.ddp_config.bucket_size
        if disable_bucketing:
            self.bucket_size = None
        self.device = device if device else torch.cuda.current_device()

        self.param_to_bucket_group = {}

        if fsdp_unit_modules is not None:
            self.fsdp_unit_modules = fsdp_unit_modules
        else:
            if self.ddp_config.data_parallel_sharding_strategy == "optim_grads_params":
                self.fsdp_unit_modules = [TransformerLayer]
            else:
                self.fsdp_unit_modules = []
        self.main_weights = True

        # Determine if we should delay the gradient reduction.
        self.is_delay_grad_reduce = self.ddp_config.data_parallel_sharding_strategy in [
            "no_shard",
            "optim",
        ]

        if self.ddp_config.data_parallel_sharding_strategy == "optim_grads_params":
            assert self.ddp_config.overlap_param_gather
        if not self.is_delay_grad_reduce:
            assert self.ddp_config.overlap_grad_reduce
        self._init_fsdp_param_and_grad_buffer()
        self._register_fsdp_hooks(self.module)

        # Delete references to weight_tensor if they exist since we don't want two parameter copies
        # if we re-mapped parameters (which happens when we use the distributed optimizer).
        # This is a temporary workaround around a TE bug that is fixed with
        # https://github.com/NVIDIA/TransformerEngine/pull/719.
        @torch.no_grad()
        def unmap_weight_tensor(m):
            if hasattr(m, 'weight_tensor'):
                m.weight_tensor = None

        self.module.apply(unmap_weight_tensor)

    def _init_fsdp_param_and_grad_buffer(self):
        if self.config.calculate_per_token_loss:
            # We don't need to scale the gradients in this case.
            gradient_scaling_factor = None
            expert_gradient_scaling_factor = None
        else:
            if self.ddp_config.average_in_collective:
                # FIXME(@jianbinc): Will fix this issue based on Parallel Folding's EDP patch MR.
                raise Exception("Not supported")
            else:
                data_parallel_world_size = self.dp_cp_group.size()
                gradient_scaling_factor = 1.0 / data_parallel_world_size
                expert_gradient_scaling_factor = 1.0 / data_parallel_world_size

        # Initialize the param and grad buffer.
        self.data_parallel_sharding_strategy = self.ddp_config.data_parallel_sharding_strategy
        self.param_to_name = {p: name for name, p in self.module.named_parameters()}
        self.param_and_grad_buffer = ParamAndGradBuffer(
            self.ddp_config,
            self.module,
            bucketing_policy=BucketingPolicy(
                suggested_bucket_size=self.bucket_size,
                fsdp_unit_modules=self.fsdp_unit_modules,
                data_parallel_sharding_strategy=self.data_parallel_sharding_strategy,
            ),
            data_parallel_group=self.dp_cp_group,
            expert_data_parallel_group=self.expt_dp_group,
            preserve_fp32_weights=self.ddp_config.preserve_fp32_weights,
            grad_reduce_in_fp32=self.ddp_config.grad_reduce_in_fp32,
            gradient_scaling_factor=gradient_scaling_factor,
            expert_gradient_scaling_factor=expert_gradient_scaling_factor,
            device=self.device,
            reset_parameters_for_meta_device_init_module=self.config.init_model_with_meta_device,
        )
        self.param_and_grad_buffer

        self.side_stream_for_buffer_copy_and_grad_accum = torch.cuda.Stream()

        # Initialize the reduce-scatter pipeline.
        self.grad_reduce_pipeline = GradReducePipeline(
            self.param_and_grad_buffer, cuda_stream=self.side_stream_for_buffer_copy_and_grad_accum
        )

        # Initialize the all-gather pipeline.
        self.all_gather_pipeline = AllGatherPipeline(self.param_and_grad_buffer)

        suggested_communication_unit_size = self.ddp_config.suggested_communication_unit_size
        if suggested_communication_unit_size is None:
            if self.data_parallel_sharding_strategy == "optim_grads_params":
                total_param_elements = 0
                total_fsdp_module = 0
                for module in self.module.modules():
                    if isinstance(module, tuple(self.fsdp_unit_modules)):
                        total_fsdp_module += 1
                        total_param_elements += sum(p.numel() for p in module.parameters())
                # The suggested size is twice the number of elements in the FSDP modules.
                # This ensures we process the current FSDP module and attempt to prefetch
                # the next FSDP module, making the flow of communication better.
                suggested_communication_unit_size = total_param_elements // total_fsdp_module * 2
            elif self.bucket_size is not None:
                suggested_communication_unit_size = self.bucket_size * 2

        self.suggested_RS_queue_capacity = suggested_communication_unit_size
        self.suggested_AG_prefetch_size = suggested_communication_unit_size

    def _register_fsdp_hooks(self, root_module):
        """Register necessary hooks for Fully Sharded Data Parallel (FSDP) execution on the model.

        This function sets up various hooks required for FSDP operations, including parameter
        resharding/unsharding and gradient handling. The registered hooks are:
            - Pre-forward hook: Unshards parameters before forward pass
            - Post-forward hook: Reshards parameters after forward pass
            - Pre-backward hook: Unshards parameters before backward pass
            - Post-backward hook: Reshards parameters and reduces gradients after backward pass

        Args:
            root_module: The PyTorch module to register FSDP hooks on

        Note:
            These hooks are essential for FSDP's memory efficiency as they manage:
            1. Dynamic parameter sharding/unsharding to reduce memory footprint
            2. Proper gradient synchronization across distributed processes
            3. Gradient accumulation for large batch training

        Returns:
            None
        """

        # Initialize module training state.
        for m in root_module.modules():
            setattr(m, "_training_state", TrainingState.IDLE)

        self.forward_pre_hooks = {}
        self.forward_hooks = {}
        self.backward_pre_hooks = {}

        """
        An FSDP unit is a module designed to manage the lifecycle of model parameters
        in Fully Sharded Data Parallel (FSDP) training. It ensures that parameters
        are only used within the module and are released immediately after
        the forward and backward computations are completed.
        This approach is crucial for efficient memory management, as releasing
        parameters too early can lead to issues if other computations depend on them.

        `optim` and `optim_grads` do not require FSDP units because they do not
        shard model parameters.
        """
        fsdp_unit_modules = self.fsdp_unit_modules

        def release_module_parameters(module, *unused):
            for param in module.parameters():
                bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                self.all_gather_pipeline.release_bucket(bucket_id)

            if not self.ddp_config.keep_fp8_transpose_cache_when_using_custom_fsdp:
                release_params_fp8_transpose_cache(module.parameters())

        def release_params_fp8_transpose_cache(params):
            for param in params:
                if is_float8tensor(param):
                    param._transpose_invalid = True
                    param._transpose = None

        def all_gather_module_parameters(
            module,
            *unused,
            prefetch=True,
            prefetch_order=PrefetchOrder.FORWARD_PASS_ORDER,
            wait_bucket_ready=True,
        ):
            ag_pipeline = self.all_gather_pipeline
            ag_pipeline.all_gather_params(
                params=list(module.parameters()),
                prefetch=prefetch,
                prefetch_order=prefetch_order,
                suggested_AG_prefetch_size=self.suggested_AG_prefetch_size,
            )
            if wait_bucket_ready:
                for param in module.parameters():
                    bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                    ag_pipeline.wait_bucket_ready(bucket_id)

        def _grad_acc(param):
            """
            Accumulate the gradient in the main_grad buffer.
            """
            group_id = self.param_and_grad_buffer.param_to_param_group[param]
            group = self.param_and_grad_buffer.parameter_groups[group_id]
            if not group.requires_grad:
                return

            overwrite_main_grad = self.ddp_config.data_parallel_sharding_strategy in [
                "optim_grads",
                "optim_grads_params",
            ]
            if overwrite_main_grad:
                if not param.grad_added_to_main_grad:
                    if param.grad is not None:
                        param.main_grad.copy_(param.grad)
                        del param.grad
                    else:
                        param.main_grad.zero_()
            else:
                if not param.grad_added_to_main_grad:
                    if param.grad is not None:
                        param.main_grad.add_(param.grad)
                        del param.grad
            # Reset the grad accumulate flag.
            param.grad_added_to_main_grad = False

        self._params_require_handle_grad = set()

        def _post_backward(module, *unused):
            if isinstance(module, tuple(fsdp_unit_modules)):
                if self.ddp_config.data_parallel_sharding_strategy == "optim_grads_params":
                    release_module_parameters(module)
                    module._training_state = TrainingState.IDLE
                param_list = list(module.parameters())
            else:
                param_list = list(module.parameters(recurse=False))

            for param in param_list:
                _grad_acc(param)
                self._params_require_handle_grad.discard(param)

            grad_reduce_every_bprop = self.ddp_config.data_parallel_sharding_strategy in [
                "optim_grads",
                "optim_grads_params",
            ]
            if grad_reduce_every_bprop or self.is_last_microbatch:
                self.grad_reduce_pipeline.reduce_gradients(
                    param_list, suggested_queue_capacity=self.suggested_RS_queue_capacity
                )

        def _pre_forward_param_unshard(
            module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
        ):
            # Unshard the parameters before the forward pass.
            input_training_state = module._training_state
            fsdp_forward_prefetch = True
            if input_training_state == TrainingState.PRE_BACKWARD:
                # In activation recomputation case, we need to cancel forward prefetch.
                fsdp_forward_prefetch = False
            else:
                module._training_state = TrainingState.FORWARD

            if isinstance(module, tuple(fsdp_unit_modules)):
                param_list = list(module.parameters())
                self.all_gather_pipeline.all_gather_params(
                    params=param_list,
                    prefetch=fsdp_forward_prefetch,
                    suggested_AG_prefetch_size=self.suggested_AG_prefetch_size,
                )
                for param in param_list:
                    bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                    self.all_gather_pipeline.wait_bucket_ready(bucket_id)
            else:
                # All-gather the parameters in every forward pass for FSDP.
                param_list = list(module.parameters(recurse=False))
                self.all_gather_pipeline.all_gather_params(
                    params=param_list,
                    prefetch=fsdp_forward_prefetch,
                    suggested_AG_prefetch_size=self.suggested_AG_prefetch_size,
                )
                for param in param_list:
                    bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                    self.all_gather_pipeline.wait_bucket_ready(bucket_id)
            return args, kwargs

        def _register_post_backward_hook(
            post_backward_hook: callable,
            module: nn.Module,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ):
            # Register the backward function to reduce gradients after the backward pass.
            # And for optim_grads_params, we need to release the parameters after the backward pass.
            if not torch.is_grad_enabled():
                return args, kwargs

            args_list, args_spec = tree_flatten(args)
            kwargs_list, kwargs_spec = tree_flatten(kwargs)
            args_kwargs_list = list(args_list) + list(kwargs_list)
            inp_tensor_indices: List[int] = []
            inp_tensors: List[torch.Tensor] = []
            for i, obj in enumerate(args_kwargs_list):
                if torch.is_tensor(obj) and obj.requires_grad:
                    inp_tensor_indices.append(i)
                    inp_tensors.append(obj)

            if len(inp_tensors) == 0:
                return args, kwargs

            inp_tensors = RegisterFSDPBackwardFunction.apply(
                functools.partial(post_backward_hook, module), *inp_tensors
            )

            for inp_tensor_idx, inp_tensor in zip(inp_tensor_indices, inp_tensors):
                args_kwargs_list[inp_tensor_idx] = inp_tensor
            args_list = args_kwargs_list[: len(args_list)]
            kwargs_list = args_kwargs_list[len(args_list) :]
            args = tree_unflatten(args_list, args_spec)
            kwargs = tree_unflatten(kwargs_list, kwargs_spec)

            return args, kwargs

        fsdp_modules = []
        for name, module in root_module.named_modules():
            if any(is_submodule(module, fsdp_module) for fsdp_module in fsdp_modules):
                continue

            if isinstance(module, tuple(fsdp_unit_modules)):
                fsdp_modules.append(module)

            self.forward_pre_hooks[f'module {name} parameter unshard'] = (
                module.register_forward_pre_hook(
                    _pre_forward_param_unshard, prepend=True, with_kwargs=True
                )
            )
            self.forward_pre_hooks[f"module {name} register post-backward hook"] = (
                module.register_forward_pre_hook(
                    functools.partial(_register_post_backward_hook, _post_backward),
                    with_kwargs=True,
                )
            )

        def _root_post_backward(*unused):
            # Make sure all the gradients are handled.
            for param in self._params_require_handle_grad:
                _grad_acc(param)

            # Reduce the remain gradients.
            grad_reduce_every_bprop = self.ddp_config.data_parallel_sharding_strategy in [
                "optim_grads",
                "optim_grads_params",
            ]
            if grad_reduce_every_bprop or self.is_last_microbatch:
                self.grad_reduce_pipeline.reduce_gradients(
                    list(self._params_require_handle_grad),
                    suggested_queue_capacity=self.suggested_RS_queue_capacity,
                )
                self.grad_reduce_pipeline.reset()

            # Reset root_pre_backward_hook_issued flag.
            self._root_pre_backward_hook_issued = False

        def _pre_backward(module: nn.Module, *unused):
            module._training_state = TrainingState.PRE_BACKWARD
            if isinstance(module, tuple(fsdp_unit_modules)):
                all_gather_module_parameters(
                    module, prefetch_order=PrefetchOrder.BACKWARD_PASS_ORDER
                )

        self._root_pre_backward_hook_issued = False

        def _root_pre_backward(module: nn.Module, *unused):
            """Marks the module's training state as 'pre_backward' before the
            backprop, this function is registered on the root module.

            This marking enables us to determine whether forward pass needs to
            perform reshard/unshard operations in activation recomputation
            scenarios.
            """
            if self._root_pre_backward_hook_issued:
                return
            self._root_pre_backward_hook_issued = True

            if self.ddp_config.data_parallel_sharding_strategy == "optim_grads_params":
                for module in root_module.modules():
                    if isinstance(module, tuple(fsdp_unit_modules)):
                        module._training_state = TrainingState.PRE_BACKWARD
                        for param in module.parameters():
                            bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                            self.all_gather_pipeline.wait_bucket_ready(bucket_id, empty_ok=True)
                            self.all_gather_pipeline.release_bucket(bucket_id)
            self._params_require_handle_grad = set()
            for param_group in self.param_and_grad_buffer.parameter_groups:
                if not param_group.requires_grad:
                    continue
                self._params_require_handle_grad |= set(param_group.params)
                for param in param_group.params:
                    param.grad_added_to_main_grad = False
            torch.autograd.Variable._execution_engine.queue_callback(_root_post_backward)

        def _post_forward(module: nn.Module, input: Any, output: Any):
            # When composing with module-hook-based activation checkpointing, the
            # post-backward hook is responsible for the reshard
            if module._training_state == TrainingState.PRE_BACKWARD:
                return output

            release_module_parameters(module)
            module._training_state = TrainingState.IDLE

            return output

        def _release_module_fp8_transpose_cache(module: nn.Module, *unused):
            release_params_fp8_transpose_cache(module.parameters(recurse=False))

        if len(fsdp_unit_modules) != 0:
            fsdp_modules = []
            for name, module in root_module.named_modules():
                if any(is_submodule(module, fsdp_module) for fsdp_module in fsdp_modules):
                    continue

                if isinstance(module, tuple(fsdp_unit_modules)):
                    fsdp_modules.append(module)
                    self.forward_hooks[f"release module {name} parameters"] = (
                        module.register_forward_hook(_post_forward, prepend=False)
                    )
                    self.backward_pre_hooks[f"all-gather module {name} parameters"] = (
                        module.register_full_backward_pre_hook(_pre_backward)
                    )
                elif not self.ddp_config.keep_fp8_transpose_cache_when_using_custom_fsdp:
                    self.forward_hooks[f"remove module {name} fp8 transpose cache"] = (
                        module.register_forward_hook(
                            _release_module_fp8_transpose_cache, prepend=False
                        )
                    )

        # Registering all models with all parameters is to handle some special cases
        # where the forward function of root_module is not called, but the forward
        # functions of these equivalent modules are called instead.
        for name, module in root_module.named_modules():
            if len(list(module.parameters())) != len(list(root_module.parameters())):
                continue

            self.backward_pre_hooks[f"{name} _root_pre_backward"] = (
                module.register_full_backward_pre_hook(_root_pre_backward)
            )
        self._root_pre_backward_hook_handle = root_module.register_full_backward_pre_hook(
            _root_pre_backward
        )

    @contextmanager
    def no_sync(self):
        """
        Context manager that turns off gradient synchronization.
        For grads shard mode there will actually always be gradient sync happening.
        """
        # FIXME: Better handling of grads shard mode and no_sync in the training loop so that
        # the code doesn't bog down developers.
        self.is_last_microbatch = False
        try:
            yield
        finally:
            self.is_last_microbatch = True

    def start_param_sync(self, *unused, force_sync: bool = False, force_dispatch: bool = False):
        """
        Initiates param sync (all-gather) communication operations for all model parameters.

        By default, when overlap_param_gather is set to True, dispatches asynchronous communication
        calls; when overlap_param_gather is set to False, calls synchronous communication
        ops. Can override this default behavior using flags below.

        Args:
            force_sync (bool, optional): force synchronous collective regardless of
                other settings.
            force_dispatch (bool, optional): force dispatch regardless of other settings.
        """
        if not force_sync and self.ddp_config.overlap_param_gather:
            # All-gather the first bucket before the forward pass.
            first_param = list(self.module.parameters())[0]
            self.all_gather_pipeline.all_gather_params(params=[first_param], prefetch=False)
        else:
            self.all_gather_pipeline.reset()
            for bucket_id in range(self.all_gather_pipeline.num_buckets):
                self.all_gather_pipeline.all_gather_bucket_and_set_items(
                    bucket_id=bucket_id, async_op=True
                )
                group = self.param_and_grad_buffer.parameter_groups[bucket_id]
                if group.model_weight_buffer is None:
                    continue

                if group.model_weight_buffer.is_data_distributed:
                    # If model weight is sharded, we wait for the all-gather to complete and
                    # then release the bucket immediately to save memory usage.
                    self.all_gather_pipeline.wait_bucket_ready(bucket_id)
            for bucket_id in range(self.all_gather_pipeline.num_buckets):
                self.all_gather_pipeline.wait_bucket_ready(bucket_id)

    def start_grad_sync(self, *unused):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        if not self.ddp_config.overlap_grad_reduce:
            if self.data_parallel_sharding_strategy == "no_shard":
                self.param_and_grad_buffer.all_reduce_gradients(
                    async_op=self.ddp_config.overlap_grad_reduce
                )
            else:
                self.param_and_grad_buffer.reduce_scatter_gradients()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        if self.ddp_config.overlap_grad_reduce:
            self.grad_reduce_pipeline.wait_for_previous_grad_reduce(0)
            self.grad_reduce_pipeline.reset()
        else:
            self.start_grad_sync()

        self.param_and_grad_buffer.update_main_grads()

        if self.ddp_config.overlap_param_gather:
            self.all_gather_pipeline.reset()

    def optimizer_named_parameters(self) -> List[Tuple[str, torch.Tensor]]:
        """
        Returns a list of tuples containing the main weights and their corresponding names
        for mixed-precision training, to be used by the optimizer for updates.

        Returns:
            List[Tuple[str, torch.Tensor]]: A list of tuples, where each tuple
                contains a main weight tensor and its corresponding name.
        """
        return self.param_and_grad_buffer.optimizer_named_parameters

    def scale_gradients(self, scaling_factor: float):
        """Scale all gradients inside the buffers by `scaling_factor`."""
        self.param_and_grad_buffer.scale_gradients(scaling_factor)

    def zero_grad_buffer(self):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.
        """
        for param in self.module.parameters():
            if param.requires_grad:
                param.grad_added_to_main_grad = False
        self.param_and_grad_buffer.zero_grad()

    def broadcast_params(self):
        """
        Syncs parameters across all DP ranks.
        """
        for param in self.module.parameters():
            is_expert_parallel = not getattr(param, 'allreduce', True)

            if is_expert_parallel:
                data_parallel_group = self.expt_dp_group
            else:
                data_parallel_group = self.dp_cp_group
            torch.distributed.broadcast(
                param.data,
                src=torch.distributed.get_global_rank(data_parallel_group, 0),
                group=data_parallel_group,
            )

    def load_state_dict(self, state_dict, strict=True):
        """
        Copies parameters and buffers from state_dict into the wrapped module and its
        descendants. If strict is True, then the keys of state_dict must exactly match
        the keys returned by this module’s state_dict() function.
        """
        if self.ddp_config.data_parallel_sharding_strategy == "optim_grads_params":
            # make a copy of the state_dict to avoid modifying the input state_dict
            state_dict = state_dict.copy()
            state_dict_extra_states = {}
            for key in list(state_dict.keys()):
                if key.endswith("_extra_state"):
                    state_dict_extra_states[key] = state_dict[key]
                    del state_dict[key]
            self.module.load_state_dict(state_dict_extra_states, strict=False)

            prefix = "module."
            buffer = self.param_and_grad_buffer
            for param_groups in buffer.parameter_groups:
                wbuf = param_groups.model_weight_buffer
                for model_param in wbuf.params:
                    if is_float8tensor(model_param):
                        fp8_meta = model_param._fp8_meta['scaling_fwd']
                        fp8_meta_index = model_param._fp8_meta_index
                        model_param._scale_inv.copy_(fp8_meta.scale_inv[fp8_meta_index])

                    param_name = f"{buffer.param_to_name[model_param]}"[len(prefix) :]
                    if param_name in state_dict:
                        if wbuf and wbuf.is_data_distributed:
                            model_param.fully_shard_param_local_shard.data.copy_(
                                state_dict[param_name]
                            )
                        else:
                            model_param.data.copy_(state_dict[param_name])
                        del state_dict[param_name]
            self.module.load_state_dict(state_dict, strict=False)
            return
        self.module.load_state_dict(state_dict, strict=strict)


class RegisterFSDPBackwardFunction(torch.autograd.Function):
    """
    Register a backward function that will be called after the backward pass
    of the model. This function is used to release the parameters after the
    backward pass.
    """

    @staticmethod
    def forward(ctx, post_backward, *inputs: torch.Tensor):
        """
        Forward pass of the RegisterFSDPBackwardFunction function.
        """
        ctx.post_backward = post_backward
        return inputs

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        """
        Backward pass of the RegisterFSDPBackwardFunction function.
        """
        ctx.post_backward()
        return (None,) + grads
