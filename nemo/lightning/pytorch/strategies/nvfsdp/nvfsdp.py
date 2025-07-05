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
# limitations under the License.import functools

import functools
import importlib
import logging
from contextlib import contextmanager
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from .utils import FSDPDistributedIndex


logger = logging.getLogger(__name__)


try:
    # Default to Megatron-LM FW.
    logger.info("Detected Megatron Core, using nvFSDP with Megatron.")
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
    from megatron.core.fp8_utils import is_float8tensor
    from megatron.core.utils import is_submodule
except ImportError:
    # Megatron-LM is not installed, use nvFSDP as a standalone module.
    logger.info("Megatron Core is not installed, nvFSDP will run without Megatron Core.")
    from .distributed_data_parallel_config import DistributedDataParallelConfig
    from .utils import is_float8tensor, is_submodule

from .param_and_grad_buffer import (
    AllGatherPipeline,
    BucketingPolicy,
    GradReducePipeline,
    ParamAndGradBuffer,
    PrefetchOrder,
    override_sharded_param_methods_with_safety_checks,
    to_local_if_dtensor,
)


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


class nvFSDP(torch.nn.Module):
    """Fully Sharded Data Parallel training.

    A distributed training wrapper that shards model parameters, gradients and optimizer
    states across data parallel workers. Integrates seamlessly with MCore's tensor
    and expert parallelism features, and in native PyTorch.

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
    - Compatible with Native PyTorch's tensor and context parallelism with DTensor
    - Automatic mixed precision training (BF16/FP8)
    - Gradient accumulation and bucketing
    - Optimized activation recompute with shard-aware communication: When recomputing
        a whole Transformer layer, gather parameters once for both the recomputation
        and backward computation
    - Compatible with MCore's distributed checkpointing, and native PyTorch.

    Args:
        module (torch.nn.Module): Underlying Torch Module.
        dist_index (FSDPDistributedIndex): FSDPDistributedIndex object containing references to the
            process groups and device meshes used by nvFSDP.
        ddp_config (DistributedDataParallelConfig): FullyShardedDataParallel configuration dataclass
            containing a variety of Megatron-derived parameters that control the behavior of nvFSDP.
        fsdp_unit_modules (List[torch.nn.Module] | List[str]): List of modules that should be treated
            as an FSDP Unit, i.e. the minimum releasable model unit. It affects the granularity of the
            communication parameter grouping and triggers aggregate collective communication in FP8
            mixed precision training.
        device (torch.device): Target device for the sharded model. Used to migrate all model parameters
            to an expected device. If init_model_with_meta_device=True, this argument is ignored.
        init_model_with_meta_device (bool): Whether to initialize model parameters in shards across all devices
            of the fsdp_group. Utilized to initialize large models that do not fit on a single device.
        sync_grads_each_step (bool): Whether to synchronize and install optimizer gradients on each
            training step. When disabled, nvFSDP will overlap reduce-scatter calls with subsequent
            compute, which improves performance and throughput when utilizing delayed optimization
            techniques such as gradient accumulation. Defaults to True for the fully_shard API.
        disable_bucketing: If true, force assign all parameters to a single bucket. If false,
            use standard bucketing policy: assign parameters to smaller buckets and all-reduce
            per bucket.

    Examples:
        >>> model = GPTModel(config)
        >>> model = nvFSDP(
        ...     model,
        ...     dist_index,
        ...     ddp_config,
        ...     fsdp_unit_modules = [TransformerLayer, LanguageModelEmbedding],
        ...     device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        ...     init_model_with_meta_device=False,
        ...     sync_grads_each_step=True,
        ...     disable_bucketing=False,
        ... )
    """

    # TODO(@jianbinc): Add HSDP (Hybrid-Sharded Data Parallel), which allows us to
    # matrix the DP group into DP-shard and DP-replicate groups to trade-off memory
    # utilization with communication overhead between DDP and FSDP.
    def __init__(
        self,
        module: torch.nn.Module,
        dist_index: FSDPDistributedIndex,
        ddp_config: DistributedDataParallelConfig = None,
        fsdp_unit_modules: Optional[List[torch.nn.Module] | List[str]] = None,
        disable_bucketing: bool = False,
        device: Optional[torch.device] = None,
        calculate_per_token_loss: bool = False,
        init_model_with_meta_device: bool = False,
        sync_grads_each_step: bool = False,
    ):
        super().__init__()
        self.device = device if device else torch.device(f"cuda:{torch.cuda.current_device()}")
        # FIXME(@jianbinc, @cspades): Conflicts with init_model_with_meta_device,
        # which avoids initializing large models on every rank. Temporary guard here.
        # Utilized to align all parameters in the model to the same device.
        self.module = module.to(self.device) if not init_model_with_meta_device else module

        # if ddp_config is not provided, use the default config
        # "optim_grads_params" is the default strategy
        if ddp_config is None:
            self.ddp_config = DistributedDataParallelConfig(
                check_for_nan_in_grad=True,
                data_parallel_sharding_strategy="optim_grads_params",
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=True,
                overlap_param_gather=True,
                average_in_collective=False,
            )
        else:
            self.ddp_config = ddp_config

        self.calculate_per_token_loss = calculate_per_token_loss
        self.init_model_with_meta_device = init_model_with_meta_device
        self.sync_grads_each_step = sync_grads_each_step

        # Check if the module contains (Megatron-Core) expert parallel parameters or DTensors.
        has_expert_parameters = self._check_module_parameter_types()

        # FSDPDistributedIndex stores the process groups and meshes used by nvFSDP.
        # If not provided, nvFSDP will default to a simple data parallel index
        # supported by torch.distributed.group.WORLD.
        self.dist_index = dist_index if dist_index is not None else FSDPDistributedIndex()

        # If Megatron Expert Parallelism is enabled, you need to provide an expt_dp_group.
        if has_expert_parameters and self.dist_index.get_expert_dp_group() is None:
            raise ValueError("[nvFSDP] Megatron Expert Parallelism is enabled, but no expt_dp_group is provided.")

        self.bucket_size = self.ddp_config.bucket_size
        if disable_bucketing:
            self.bucket_size = None

        # Parse FSDP unit modules. If given a list of strings, import the classes.
        self.fsdp_unit_modules = (
            [
                (self._import_classes_from_paths(cls_path) if isinstance(cls_path, str) else cls_path)
                for cls_path in fsdp_unit_modules
            ]
            if fsdp_unit_modules is not None
            else []
        )

        # Determine if we should delay the gradient reduction.
        self.is_delay_grad_reduce = self.ddp_config.data_parallel_sharding_strategy in [
            "no_shard",
            "optim",
        ]
        if self.ddp_config.data_parallel_sharding_strategy == "optim_grads_params":
            # Default to overlapped NCCL communication when fully-sharding.
            self.ddp_config.overlap_param_gather = True
            self.ddp_config.overlap_grad_reduce = True
        if not self.is_delay_grad_reduce:
            # Gradient reduce-scatter must be overlapped when using sharding optimizer and gradients.
            assert self.ddp_config.overlap_grad_reduce

        self._init_fsdp_param_and_grad_buffer()
        self._register_fsdp_hooks(self.module)

        self.is_param_fsdp_distributed = False
        self._replace_param_with_distributed_if_needed()

    def _check_module_parameter_types(self):
        """
        Check if the module parameters include special parameters
        such as Megatron-Core Expert Parallel (EP/EXPT) parameters.
        """
        expert_params = False
        for _, param in self.module.named_parameters():
            if not getattr(param, "allreduce", True):
                expert_params = True
            if expert_params:
                # Detected. No need to check further.
                break
        return expert_params

    def _init_fsdp_param_and_grad_buffer(self):
        if self.calculate_per_token_loss:
            # We don't need to scale the gradients in this case.
            gradient_scaling_factor = None
            expert_gradient_scaling_factor = None
        else:
            if self.ddp_config.average_in_collective:
                # FIXME(@jianbinc): Will fix this issue based on Parallel Folding's EDP patch MR.
                raise Exception("Not supported")
            else:
                data_parallel_world_size = self.dist_index.get_fsdp_group().size()
                gradient_scaling_factor = 1.0 / data_parallel_world_size
                expert_gradient_scaling_factor = 1.0 / data_parallel_world_size

        # Initialize the param and grad buffer.
        self.data_parallel_sharding_strategy = self.ddp_config.data_parallel_sharding_strategy
        self.param_and_grad_buffer = ParamAndGradBuffer(
            self.ddp_config,
            self.module,
            bucketing_policy=BucketingPolicy(
                suggested_bucket_size=self.bucket_size,
                fsdp_unit_modules=self.fsdp_unit_modules,
                data_parallel_sharding_strategy=self.data_parallel_sharding_strategy,
            ),
            dist_index=self.dist_index,
            preserve_fp32_weights=self.ddp_config.preserve_fp32_weights,
            grad_reduce_in_fp32=self.ddp_config.grad_reduce_in_fp32,
            gradient_scaling_factor=gradient_scaling_factor,
            expert_gradient_scaling_factor=expert_gradient_scaling_factor,
            device=self.device,
            reset_parameters_for_meta_device_init_module=self.init_model_with_meta_device,
        )
        self.param_to_name = {p: name for name, p in self.module.named_parameters()}
        self.raw_param = dict(self.module.named_parameters())

        # Initialize a gradient buffer and accumulation stream for the GradReducePipeline.
        self.side_stream_for_buffer_copy_and_grad_accum = torch.cuda.Stream()

        # Initialize the reduce-scatter pipeline.
        self.grad_reduce_pipeline = GradReducePipeline(
            self.param_and_grad_buffer,
            cuda_stream=self.side_stream_for_buffer_copy_and_grad_accum,
        )

        # Initialize the all-gather pipeline.
        self.all_gather_pipeline = AllGatherPipeline(self.param_and_grad_buffer)

        # Set the suggested communication unit size for reduce-scatter and all-gather pipelines.
        suggested_communication_unit_size = self.ddp_config.suggested_communication_unit_size or 1_000_000_000
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
                suggested_communication_unit_size = self.bucket_size

            # Cap to 1B elements.
            suggested_communication_unit_size = max(1_000_000_000, suggested_communication_unit_size)

        self.suggested_RS_queue_capacity = suggested_communication_unit_size
        self.suggested_AG_prefetch_size = suggested_communication_unit_size // 2

        if self.data_parallel_sharding_strategy == "optim_grads_params":
            override_sharded_param_methods_with_safety_checks(self.module.parameters(), self.all_gather_pipeline)

    def _import_classes_from_paths(self, class_paths: List[str]):
        """Helper function to import classes from string paths."""
        classes = []
        for path in class_paths:
            module_path, class_name = path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            classes.append(cls)
        return classes

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
            """
            All-gather all module parameters.
            """
            if self.data_parallel_sharding_strategy == "no_shard":
                return
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

            Utilizes the patched main_grad property of the parameter to allocate
            or fetch the main gradient bucket for the parameter.
            """
            group_id = self.param_and_grad_buffer.param_to_param_group[param]
            group = self.param_and_grad_buffer.parameter_groups[group_id]
            if not group.requires_grad:
                return

            # Sharded Gradient Buffer
            gbuf = group.main_grad_buffer
            if gbuf.is_data_distributed:
                if not param.grad_added_to_main_grad:
                    # Get `main_grad` will allocate bucket, check that the currently
                    # used main_grad buffer does not exceed the scope of two FSDP Unit
                    # Modules, i.e., the buffer limit imposed by double-buffer allocator.
                    if self.ddp_config.fsdp_double_buffer:
                        self.grad_reduce_pipeline._enforce_double_buffer_limit([group_id])

                    if param.grad is not None:
                        # Copy the gradient into the allocated main gradient bucket.
                        param.main_grad.copy_(to_local_if_dtensor(param.grad))
                        del param.grad
                    else:
                        param.main_grad.zero_()
            # Unsharded Gradient Buffer
            else:
                if not param.grad_added_to_main_grad:
                    if param.grad is not None:
                        # Add the gradient into the allocated main gradient bucket.
                        param.main_grad.add_(to_local_if_dtensor(param.grad))
                        del param.grad
            # Reset the grad accumulate flag.
            param.grad_added_to_main_grad = False

        self._params_require_handle_grad = set()

        def _post_backward(module, *unused):
            """
            Deallocate the module parameters after the backward pass,
            and reduce-scatter the gradients before the optimizer step.
            """
            if isinstance(module, tuple(fsdp_unit_modules)):
                if self.ddp_config.data_parallel_sharding_strategy == "optim_grads_params":
                    # Deallocate the module parameters after the backward pass,
                    # because we have our data-parallel gradients computed.
                    release_module_parameters(module)
                    module._training_state = TrainingState.IDLE
                param_list = list(module.parameters())
            else:
                param_list = list(module.parameters(recurse=False))

            # If the parameter is shared, we do not accumulate gradients
            # here, as the gradients will be accumulated in the
            # root post-backward hook.
            param_list = [p for p in param_list if not getattr(p, "_is_shared", False)]

            # Write computed gradients into the allocated main gradient bucket for reduce-scatter.
            for param in param_list:
                _grad_acc(param)
                self._params_require_handle_grad.discard(param)

            grad_reduce_every_bprop = self.ddp_config.data_parallel_sharding_strategy in [
                "optim_grads",
                "optim_grads_params",
            ]
            if grad_reduce_every_bprop or getattr(self, "is_last_microbatch", False):
                # Reduce-scatter the gradients asynchronously before the optimizer step.
                # Requires calling finish_grad_sync() to wait for the reduce-scatter to complete.
                self.grad_reduce_pipeline.reduce_gradients(
                    param_list,
                    suggested_queue_capacity=self.suggested_RS_queue_capacity,
                )

        def _pre_forward_param_unshard(module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
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
                # All-gather the shallow parameters in every forward pass for modules
                # that are not FSDP units. Do not recurse unless absolutely necessary,
                # to allocate as little memory as possible for this forward pass.
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
            """
            Pre-forward hook utilized to attach a gradient reduction post-backward hook to the module.
            """
            # Register the backward function to reduce gradients after the backward pass.
            # And for optim_grads_params, we need to release the parameters after the backward pass.
            if not torch.is_grad_enabled():
                return args, kwargs

            # Preprocess the input arguments.
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

            """
            Bootstrapped identity autograd function that attaches a post-backward
            "hook" to the module to trigger model resharding / deallocation and
            gradient reduce-scatter immediately after the module backward pass has
            completed to deallocate this layer's model and gradient memory before
            the subsequent backward pass.
            """
            inp_tensors = RegisterFSDPBackwardFunction.apply(
                functools.partial(post_backward_hook, module), *inp_tensors
            )

            # Post-process the input arguments for input into the module.
            for inp_tensor_idx, inp_tensor in zip(inp_tensor_indices, inp_tensors):
                args_kwargs_list[inp_tensor_idx] = inp_tensor
            args_list = args_kwargs_list[: len(args_list)]
            kwargs_list = args_kwargs_list[len(args_list) :]
            args = tree_unflatten(args_list, args_spec)
            kwargs = tree_unflatten(kwargs_list, kwargs_spec)

            # Return original input to the module forward pass.
            return args, kwargs

        def _root_post_backward(*unused):
            # Make sure all the gradients are handled.
            for param in self._params_require_handle_grad:
                _grad_acc(param)

            # Reduce the remaining gradients.
            grad_reduce_every_bprop = self.ddp_config.data_parallel_sharding_strategy in [
                "optim_grads",
                "optim_grads_params",
            ]
            if grad_reduce_every_bprop or getattr(self, "is_last_microbatch", False):
                self.grad_reduce_pipeline.reduce_gradients(
                    list(self._params_require_handle_grad),
                    suggested_queue_capacity=self.suggested_RS_queue_capacity,
                )
                self.grad_reduce_pipeline.reset()

            # If sync_grads_each_step is enabled, we automatically synchronize gradients
            # so the user does not have to call finish_grad_sync() manually. However,
            # this will reduce training performance when using delayed optimization
            # techniques such as gradient accumulation, because asynchronous gradient
            # reduce-scatter calls can be overlapped with subsequent compute.
            if self.sync_grads_each_step:
                self.finish_grad_sync()

            # Reset root_pre_backward_hook_issued flag.
            self._root_pre_backward_hook_issued = False

        def _pre_backward(module: nn.Module, *unused):
            """
            Sub-module pre-backward hook to all-gather the module parameters
            before the backward pass.
            """
            # Set the module's training state to PRE_BACKWARD to skip resharding
            # and unsharding operations when performing activation recomputation
            # / gradient checkpointing.
            module._training_state = TrainingState.PRE_BACKWARD
            if isinstance(module, tuple(fsdp_unit_modules)):
                # All-gather / unshard the module parameters before the backward pass.
                all_gather_module_parameters(module, prefetch_order=PrefetchOrder.BACKWARD_PASS_ORDER)

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
                        # Set PRE_BACKWARD state to skip resharding and unsharding operations
                        # when performing activation recomputation / gradient checkpointing.
                        module._training_state = TrainingState.PRE_BACKWARD
                        # Deallocate all model parameter buckets before the backwards pass
                        # to avoid memory spikes due to concurrently allocated buckets.
                        for param in module.parameters():
                            bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                            self.all_gather_pipeline.wait_bucket_ready(bucket_id, empty_ok=True)
                            self.all_gather_pipeline.release_bucket(bucket_id)
            # Track parameters that require gradient reduction and optimization.
            self._params_require_handle_grad = set()
            for param_group in self.param_and_grad_buffer.parameter_groups:
                if not param_group.requires_grad:
                    continue
                self._params_require_handle_grad |= set(param_group.params)
                for param in param_group.params:
                    param.grad_added_to_main_grad = False
            # Queue the root post-backward hook to reduce leftover gradients after the backward pass.
            torch.autograd.Variable._execution_engine.queue_callback(_root_post_backward)

        def _post_forward(module: nn.Module, input: Any, output: Any):
            # When composed with module-hook-based activation recomputation, the
            # post-backward hook is responsible for resharding the module parameters
            # after the forward pass. Skip resharding the module parameters in this case.
            if module._training_state == TrainingState.PRE_BACKWARD:
                # Skip weight deallocation until the backward pass is complete
                # during activation recomputation / gradient checkpointing.
                return output

            # Release the module parameters after the forward pass to save memory.
            release_module_parameters(module)
            module._training_state = TrainingState.IDLE

            return output

        def _release_module_fp8_transpose_cache(module: nn.Module, *unused):
            release_params_fp8_transpose_cache(module.parameters(recurse=False))

        def create_custom_backward_hook(module, custom_backward_handler):
            """
            Creates a custom backward hook via attaching a gradient-triggered hook
            to the output tensor(s) of a module during a post-forward hook.
            """

            def forward_hook(_module, inputs, output):
                output_list = []

                # Post-process forward output.
                if isinstance(output, torch.Tensor):
                    output_list = [output]
                elif isinstance(output, (tuple, list)):
                    output_list = [t for t in output if isinstance(t, torch.Tensor)]

                # Register pre-backward hook on the output tensor(s). This hook
                # will trigger immediately after the gradients of the output
                # tensor(s) have been computed.
                torch.autograd.graph.register_multi_grad_hook(
                    output_list,
                    lambda grads: custom_backward_handler(_module, grads),
                    mode="any",
                )
                return output

            # Register the post-forward hook that attaches the custom backward hook
            # on the output tensor(s).
            return module.register_forward_hook(forward_hook)

        fsdp_modules = []
        for name, module in root_module.named_modules():
            # Skip if the module is already registered in fsdp_modules.
            if any(is_submodule(module, fsdp_module) for fsdp_module in fsdp_modules):
                continue

            # Register the forward pre-hook to unshard parameters before the forward pass.
            # If we are not sharding anything, we do not have a model weight buffer and thus
            # have nothing to all-gather / un-shard.
            if self.ddp_config.data_parallel_sharding_strategy != "no_shard":
                self.forward_pre_hooks[f"module {name} parameter unshard"] = module.register_forward_pre_hook(
                    _pre_forward_param_unshard, prepend=True, with_kwargs=True
                )

            if isinstance(module, tuple(fsdp_unit_modules)):
                fsdp_modules.append(module)
                # Register the forward post-hook to reshard FSDP unit module parameters
                # after the forward pass, except when recomputing forward activations,
                # in which case we skip resharding for the subsequent backward pass.
                self.forward_hooks[f"release module {name} parameters"] = module.register_forward_hook(
                    _post_forward, prepend=False
                )

                # Register the backward pre-hook to unshard FSDP unit module parameters
                # immediately before the backward pass via attaching a gradient-triggered
                # hook to the output tensor(s) of a module during a post-forward hook.
                self.backward_pre_hooks[f"all-gather module {name} parameters"] = create_custom_backward_hook(
                    module, _pre_backward
                )
            elif (
                not self.ddp_config.keep_fp8_transpose_cache_when_using_custom_fsdp
                and self.ddp_config.data_parallel_sharding_strategy == "optim_grads_params"
            ):
                # Register the forward post-hook to release FP8 transpose cache
                # after the forward pass for non-FSDP unit modules.
                # NOTE: We only need to remove the transpose cache in parameter
                # sharding strategy.
                self.forward_hooks[f"remove module {name} fp8 transpose cache"] = module.register_forward_hook(
                    _release_module_fp8_transpose_cache, prepend=False
                )

            # Register the post-backward hook to deallocate model parameters and
            # reduce-scatter gradients immediately after the module backward pass
            # has completed to conserve memory for the subsequent backward pass.
            self.forward_pre_hooks[f"module {name} register post-backward hook"] = module.register_forward_pre_hook(
                functools.partial(_register_post_backward_hook, _post_backward),
                with_kwargs=True,
            )

        # Register root module pre- and post-backward hooks in cases where the
        # forward function of root module is not called, but rather the forward
        # function of the root module from named_modules() is called instead.
        for name, module in root_module.named_modules():
            if len(list(module.parameters())) != len(list(root_module.parameters())):
                # Only attach to root sub-module.
                continue
            # Add a pre-backward hook to reshard / deallocate model parameters prior to the backward pass.
            # Furthermore, add a gradient-triggered post-backward hook to reduce-scatter leftover gradients.
            self.backward_pre_hooks[f"{name} _root_pre_backward"] = create_custom_backward_hook(
                module, _root_pre_backward
            )
        self._root_pre_backward_hook_handle = create_custom_backward_hook(module, _root_pre_backward)

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

    def get_distributed_index(self) -> FSDPDistributedIndex:
        """
        Get the distributed environment of nvFSDP, which contains references
        to the process groups and device meshes used by nvFSDP.
        """
        return self.dist_index

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
        if self.ddp_config.data_parallel_sharding_strategy == "optim_grads_params":
            self._replace_param_with_raw_if_needed()

        if not force_sync and self.ddp_config.overlap_param_gather:
            # All-gather the first bucket before the forward pass.
            first_param = list(self.module.parameters())[0]
            self.all_gather_pipeline.all_gather_params(params=[first_param], prefetch=False)
        else:
            self.synchronize_param_gather()
            for bucket_id in range(self.all_gather_pipeline.num_buckets):
                self.all_gather_pipeline.all_gather_bucket_and_set_items(bucket_id=bucket_id, async_op=True)
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
                self.param_and_grad_buffer.all_reduce_gradients(async_op=self.ddp_config.overlap_grad_reduce)
            else:
                self.param_and_grad_buffer.reduce_scatter_gradients()

    def synchronize_param_gather(self):
        """
        Synchronize parameter all-gather operations for all model parameters.
        """
        self.all_gather_pipeline.reset()

    def synchronize_gradient_reduce(self):
        """
        Synchronize gradient reduce-scatter operations for all model gradients.
        """
        if self.ddp_config.overlap_grad_reduce:
            # Asynchronous reduce-scatter from overlap_grad_reduce=True,
            # i.e. when sharding optimizer and gradients.
            self.grad_reduce_pipeline.wait_for_previous_grad_reduce(0)
            self.grad_reduce_pipeline.reset()
        else:
            # Synchronous gradient all-reduce when sharding optimizer state or not sharding.
            self.start_grad_sync()

    def attach_grad_to_optimizer_state(self):
        """
        Attach gradients to optimizer named parameters
        in preparation for optimizer.step().
        """
        self.param_and_grad_buffer.update_main_grads()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients. Call prior to the optimization step to resolve
        asynchronous gradient reductions.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        # Synchronize gradient reduce-scatter operations for all model gradients.
        self.synchronize_gradient_reduce()

        # Once the gradients have been reduced and scattered into main_grad_buffer,
        # update the gradients for all buffered weights in optimizer_named_parameters.
        self.attach_grad_to_optimizer_state()

        # Synchronize parameter all-gather operations for all model parameters,
        # which are triggered during the backward pass for FSDP.
        if self.ddp_config.overlap_param_gather:
            self.synchronize_param_gather()

        # Before the optimizer.step(), replace raw module parameters with distributed
        # optimizer named parameters for distributed optimization.
        self._replace_param_with_distributed_if_needed()

    def _replace_param_with_distributed_if_needed(self):
        if self.is_param_fsdp_distributed:
            return
        self.is_param_fsdp_distributed = True

        pg_buffer = self.param_and_grad_buffer
        fsdp_params = dict(pg_buffer.optimizer_named_parameters)
        for name, _ in self.module.named_parameters():
            assert name in fsdp_params, f"Parameter {name} not found in FSDP parameters."
            dist_param = fsdp_params[name]
            _replace_module_parameter(self.module, name, dist_param)

        # Handle shared weights
        self._reestablish_shared_weights(self.raw_param, fsdp_params)

    def _replace_param_with_raw_if_needed(self):
        if not self.is_param_fsdp_distributed:
            return
        self.is_param_fsdp_distributed = False

        for name, _ in self.module.named_parameters():
            assert name in self.raw_param, f"Raw parameter {name} not found in module."
            _replace_module_parameter(self.module, name, self.raw_param[name])

        # Handle shared weights
        pg_buffer = self.param_and_grad_buffer
        fsdp_params = dict(pg_buffer.optimizer_named_parameters)
        self._reestablish_shared_weights(fsdp_params, self.raw_param)

    def _reestablish_shared_weights(self, old_params, new_params):
        """
        Reestablishes shared (tied) weights in a PyTorch module after parameter replacement.

        When iterating over `named_parameters()`, PyTorch skips parameters that are shared
        via weight-tying (e.g., `lm_head.weight` referencing `wte.weight`). After replacing
        parameters, these shared weights become independent, causing previously hidden
        parameters to appear in the parameter list. This function restores the original
        shared structure by ensuring parameters that were previously tied remain shared.

        Args:
            old_params (dict): Mapping from parameter names to original parameter tensors.
            new_params (dict): Mapping from parameter names to new parameter tensors.
        """
        for name, param in self.module.named_parameters():
            if name in new_params:
                # Parameter was explicitly replaced; nothing to do.
                continue

            # Attempt to find the corresponding shared parameter in old_params.
            shared_param = None
            for old_name, old_weight in old_params.items():
                # Found a shared parameter; get the new version.
                if id(param) == id(old_weight):
                    shared_param = new_params.get(old_name)
                    break
            assert shared_param is not None, f"Parameter {name} not found in new parameters or as a shared weight."

            # Replace the module parameter with the restored shared parameter.
            _replace_module_parameter(self.module, name, shared_param)
            setattr(shared_param, "_is_shared", True)  # Mark as shared

    def scale_gradients(self, scaling_factor: float):
        """Scale all gradients inside the buffers by `scaling_factor`."""
        self.param_and_grad_buffer.scale_gradients(scaling_factor)

    def zero_grad_buffer(self):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration alongside optimizer.zero_grad().
        """
        for param in self.module.parameters():
            if param.requires_grad:
                param.grad_added_to_main_grad = False
        self.param_and_grad_buffer.zero_grad()

    def install_optimized_model_weights(self):
        """
        Copies optimized parameter values into the model training parameters
        managed by nvFSDP. Should be called after the optimizer.step().
        """
        self.param_and_grad_buffer.copy_main_weights_to_model_weights()

    def broadcast_params(self):
        """
        Syncs parameters across all DP ranks.
        """
        for param in self.module.parameters():
            is_expert_parallel = not getattr(param, "allreduce", True)

            if is_expert_parallel:
                # Broadcast across the EP-DP group.
                data_parallel_group = self.dist_index.get_expert_dp_group()
            else:
                # Broadcast across the FSDP-CP group.
                data_parallel_group = self.dist_index.get_fsdp_group()
            torch.distributed.broadcast(
                param.data,
                src=torch.distributed.get_global_rank(data_parallel_group, 0),
                group=data_parallel_group,
            )

    def forward(self, *inputs, **kwargs):
        """
        Wrapped forward pass of the model managed by FSDP.
        """
        self._replace_param_with_raw_if_needed()
        with torch.autograd.profiler.record_function("CustomFSDP.forward"):
            # Call the forward pass of the wrapped module.
            output = self.module.forward(*inputs, **kwargs)
            return output


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


def _replace_module_parameter(module, name, new_param):
    """
    Replace a module's parameter with a new parameter, preserving the hierarchy.
    """
    parts = name.split(".")
    parent = module
    for part in parts[:-1]:  # Navigate to parent module
        parent = getattr(parent, part)

    # Replace the parameter
    setattr(parent, parts[-1], new_param)
