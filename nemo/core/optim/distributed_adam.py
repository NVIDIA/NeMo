# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import collections
import itertools

import torch
from apex.contrib.optimizers.distributed_fused_adam import (
    DistributedFusedAdam,
    _coalescing_manager,
    _coalescing_manager_append_work,
    _disable_pre_forward_hook,
)
from megatron.core import parallel_state


def _str_to_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).strip().lower()
    if name in ('', 'none'):
        return torch.float32
    elif name in ('torch.float32', 'float32', 'float', 'fp32', '32'):
        return torch.float32
    elif name in ('torch.float16', 'float16', 'half', 'fp16', '16'):
        return torch.float16
    elif name in ('torch.bfloat16', 'bfloat16', 'bf16'):
        return torch.bfloat16
    else:
        raise ValueError(f'unsupported dtype ({dtype})')


class MegatronDistributedFusedAdam(DistributedFusedAdam):
    """Wrapper class that supports NeMo-Megatron optimizations

    When O2-style optimizations are enabled, gradients are accumulated
    into the main_grad buffer instead of the grad buffer.

    """

    def __init__(self, params, disable_distributed_parameters=False, **kwargs):

        # Initialize process groups
        if 'process_group' not in kwargs and not parallel_state.is_unitialized():
            kwargs['process_group'] = parallel_state.get_data_parallel_group()
        if disable_distributed_parameters:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            self_groups = [torch.distributed.new_group(ranks=[i]) for i in range(world_size)]
            kwargs['distributed_process_group'] = self_groups[rank]
            kwargs['redundant_process_group'] = kwargs['process_group']

        # Make sure dtypes are in right type
        for keyword in ('dtype', 'grad_sync_dtype', 'param_sync_dtype'):
            if keyword in kwargs:
                kwargs[keyword] = _str_to_dtype(kwargs[keyword])

        # Make sure params are in consistent format (list of param group dicts)
        param_groups = list(params)
        assert param_groups
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        # Check if explicit FP32 optimizer is needed
        self._fp32_optim = None
        distopt_param_groups = param_groups
        dtype = kwargs['dtype'] if 'dtype' in kwargs else torch.float32
        grad_sync_dtype = kwargs['grad_sync_dtype'] if 'grad_sync_dtype' in kwargs else dtype
        needs_fp32_optimizer = any(
            getattr(param, '_with_fp32_optimizer', False)
            for param in itertools.chain.from_iterable(param_group['params'] for param_group in param_groups)
        )
        if (dtype != torch.float32 or grad_sync_dtype != torch.float32) and needs_fp32_optimizer:

            # Find params that require explicit FP32 optimizer
            distopt_param_groups = []
            fp32_param_groups = []
            self._fp32_optim_main_params = collections.OrderedDict()
            for param_group in param_groups:
                distopt_param_group = {key: val for key, val in param_group.items() if key != 'params'}
                distopt_param_group['params'] = []
                fp32_param_group = {key: val for key, val in param_group.items() if key != 'params'}
                fp32_param_group['params'] = []
                for model_param in param_group['params']:
                    if getattr(model_param, '_with_fp32_optimizer', False):
                        main_param = model_param.detach().clone().float()
                        fp32_param_group['params'].append(main_param)
                        self._fp32_optim_main_params[model_param] = main_param
                    else:
                        distopt_param_group['params'].append(model_param)
                distopt_param_groups.append(distopt_param_group)
                fp32_param_groups.append(fp32_param_group)

            # Construct explicit FP32 optimizer
            adamw_kwargs = {}
            for name in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad'):
                if name in kwargs:
                    adamw_kwargs[name] = kwargs[name]
            self._fp32_optim = torch.optim.AdamW(fp32_param_groups, **adamw_kwargs)
            self._fp32_optim_grad_sync_needed = True

        # Construct distributed optimizer
        super().__init__(distopt_param_groups, **kwargs)

    def _make_post_backward_hook(self, param, param_group_id, param_id):
        def hook(*unused):
            if getattr(param, '_pre_forward_hook_is_enabled', False):
                raise RuntimeError(
                    'A parameter called its post-backward hook '
                    'before its pre-forward hook. '
                    'Please manually interact with the parameter '
                    'before the forward pass (e.g. by calling data_ptr) '
                    'or run DistributedFusedAdam with overlap_param_sync=False.'
                )
            with self._lock:
                need_to_initialize = 'fragments' not in self.state[param]
                if need_to_initialize:
                    self._init_param_state(param, param_group_id, param_id)
                if self.greedy_grad_copy and not getattr(param, '_disable_greedy_grad_copy', False):
                    self._grad_copy(param)
                    if self.overlap_grad_sync and not getattr(param, '_disable_overlap_grad_sync', False):
                        self._try_start_bucket_grad_sync(
                            params=[param], ignore_last_bucket=need_to_initialize,
                        )

        return hook

    def _filter_distopt_params(self, params):
        if self._fp32_optim is None:
            return params
        if params is None:
            return None
        if isinstance(params, torch.Tensor):
            params = [params]
        return filter(lambda param: param not in self._fp32_optim_main_params, params)

    def parameters(self, with_fp32_optim_params=False):
        if with_fp32_optim_params and self._fp32_optim is not None:
            return itertools.chain(super().parameters(), self._fp32_optim_main_params.keys())
        else:
            return super().parameters()

    def init_params(self, params=None):
        super().init_params(self._filter_distopt_params(params))

    def init_params_bucket(self, params):
        super().init_params_bucket(self._filter_distopt_params(params))

    def try_grad_sync(self, params):
        params = self._filter_distopt_params(params)
        params = [p for p in params if not getattr(p, '_disable_greedy_grad_copy', False)]
        params = [p for p in params if not getattr(p, '_disable_overlap_grad_sync', False)]
        for p in params:
            self._grad_copy(p)
        self._try_start_bucket_grad_sync(params=params)

    def _try_start_bucket_param_sync(self, params=None):
        super()._try_start_bucket_param_sync(self._filter_distopt_params(params))

    def _fp32_optim_grad_sync(self):
        if self._fp32_optim is None or not self._fp32_optim_grad_sync_needed:
            return
        for model_param, main_param in self._fp32_optim_main_params.items():
            if model_param.grad is not None:
                main_param.grad += model_param.grad.detach()
        with _coalescing_manager(self.process_group, self.device, async_ops=True) as cm:
            for main_param in self._fp32_optim_main_params.values():
                _coalescing_manager_append_work(
                    cm,
                    torch.distributed.all_reduce(
                        main_param.grad, op=torch.distributed.ReduceOp.AVG, group=self.process_group, async_op=True,
                    ),
                )
        cm.wait()
        self._fp32_optim_grad_sync_needed = False

    def zero_grad(self, *args, **kwargs):
        super().zero_grad(*args, **kwargs)

        # Reset grads for explicit FP32 optimizer
        if self._fp32_optim is not None:
            self._fp32_optim_grad_sync_needed = True
            self._fp32_optim.zero_grad(set_to_none=False)
            for model_param, main_param in self._fp32_optim_main_params.items():
                if main_param.grad is None:
                    main_param.grad = torch.zeros_like(main_param)
                if model_param.grad is not None:
                    model_param.grad.zero_()
                model_param.main_grad = main_param.grad

        # Reset main grads
        if self.contiguous_grad_buffer:
            for param in self.parameters():
                with _disable_pre_forward_hook(param):
                    param.main_grad = self.grad_buffer_view(param)

    def grad_norm(self, parameters=None, norm_type=2.0, force=False):
        assert norm_type == 2

        if parameters is not None:
            # Make sure we can access iterable multiple times
            parameters = list(parameters)

        # Compute grad norm
        if force or self._grad_norm is None:

            # Compute norm of local gradients for distributed optimizer
            grad_norm_sq = self._local_grad_norm(
                parameters=self._filter_distopt_params(parameters), norm_type=norm_type,
            )
            if self.redundant_size > 1:
                grad_norm_sq /= self.redundant_size

            # Compute norm of local gradients for explicit FP32 optimizer
            if self._fp32_optim is not None:
                self._fp32_optim_grad_sync()
                if parameters is None:
                    for main_param in self._fp32_optim_main_params.values():
                        grad_norm_sq += torch.linalg.norm(main_param.grad) ** 2 / self.process_group_size
                else:
                    for model_param in parameters:
                        if model_param in self._fp32_optim_main_params:
                            main_param = self._fp32_optim_main_params[model_param]
                            grad_norm_sq += torch.linalg.norm(main_param.grad) ** 2 / self.process_group_size

            # Sum over all procs to get grad norm
            torch.distributed.all_reduce(
                grad_norm_sq, op=torch.distributed.ReduceOp.SUM,
            )
            self._grad_norm = grad_norm_sq.sqrt()

        # Use cached grad norm
        return super().grad_norm()

    def step(self, closure=None, *, grad_scaler=None):

        # Apply distributed optimizer
        loss = super().step(closure=closure, grad_scaler=grad_scaler)

        if self._fp32_optim is not None:

            # Handle grad scaling
            if grad_scaler is not None:
                scaler_state = grad_scaler._per_optimizer_states[id(self)]
                for _, found_inf in scaler_state['found_inf_per_device'].items():
                    if found_inf.item():
                        return loss

            # Update learning rate
            for distopt_group, fp32_optim_group in zip(self.param_groups, self._fp32_optim.param_groups):
                fp32_optim_group['lr'] = distopt_group['lr']

            # Apply explicit FP32 optimizer
            self._fp32_optim_grad_sync()
            for main_param in self._fp32_optim_main_params.values():
                main_param.grad *= self._grad_scale
            self._fp32_optim.step()
            for model_param, main_param in self._fp32_optim_main_params.items():
                model_param.detach().copy_(main_param.detach())

        return loss

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if self._fp32_optim is not None and state_dict is not None:
            state_dict['fp32_optim'] = self._fp32_optim.state_dict()
            state_dict['fp32_optim_fp32_params'] = list(self._fp32_optim_main_params.values())
        return state_dict

    def sharded_state_dict(self, model_sharded_state_dict=None):
        assert (
            model_sharded_state_dict is not None
        ), f'{self.__class__.__name__}.state_dict_for_save_checkpoint requires passing model checkpoint with ShardedTensors'

        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict, chain.from_iterable(g['params'] for g in self.param_groups)
        )

        # Convert state
        state_dict = self.state_dict(gather_on_root=False)
        state_dict = self.optim_state_to_sharding_state(state_dict, id_to_sharded_param_map)

        # Fp32 optimizer
        self.add_fp32_sharding_state(state_dict, model_sharded_state_dict)

        return state_dict

    def load_state_dict(self, state_dict):
        if 'gathered_states' in state_dict:
            # regular checkpoint
            if self._fp32_optim is not None and 'fp32_optim' in state_dict:
                self._fp32_optim.load_state_dict(state_dict['fp32_optim'])
                del state_dict['fp32_optim']
                for old_main_param, new_main_param in zip(
                    self._fp32_optim_main_params.values(), state_dict['fp32_optim_fp32_params']
                ):
                    old_main_param.copy_(new_main_param.detach())
                del state_dict['fp32_optim_fp32_params']
            return super().load_state_dict(state_dict)

        # Resuming from non-distributed optimizer case. TODO: handle this in more general way (if we want to support it in NeMo)
        if 'optimizer' in state_dict:
            state_dict['param_groups'] = merge(state_dict['optimizer']['param_groups'], state_dict['param_groups'])

        # Copy `step` from non-distributed optimizer. TODO: handle this in more general way (if we want to support it in NeMo)
        if 'step' not in state_dict['state'] and 'step' in state_dict['param_groups'][0]:
            state_dict['state']['step'] = state_dict['param_groups'][0].pop('step')

        # Deallocate existing buckets state (will be overwritten) to free some
        # cuda memory and aggregate bucket state from fragments on cpu
        for bucket in self.state['buckets']:
            for field in ('exp_avg_shard', 'exp_avg_sq_shard', 'params_shard', 'param_remainders_shard'):
                field_tensor = getattr(bucket, field)
                if field_tensor is None:
                    setattr(bucket, field, None)
                else:
                    assert field_tensor.is_cuda, (field, field_tensor.device)
                    # Aggregate bucket states on cpu
                    setattr(bucket, field, torch.empty_like(field_tensor, device='cpu'))

        optim_state = state_dict['state']
        assert optim_state['buckets'] == self.state['buckets'], (
            'When loading from distributed checkpoint, buckets should be' ' wrapped with LocalNonpersitentObject'
        )

        # TODO: maybe run consistency validation here
        # Copies relevant fragments data from checkpoint into the buckets.
        # Check `make_sharded_fragment_data` docs for further explanation
        # (here we "reverse" that method).
        for param_id, fragments_data in optim_state['fragments_local_data'].items():
            for fragment_id, fragment_data in fragments_data.items():
                fragment_data: torch.Tensor
                fragment: DistributedFusedAdam.ParameterFragment = optim_state[param_id]['fragments'][fragment_id]
                bucket = optim_state['buckets'][fragment.bucket_id]

                bucket.exp_avg_shard[slice(*fragment.shard_range)] = fragment_data['exp_avg_shard']
                bucket.exp_avg_sq_shard[slice(*fragment.shard_range)] = fragment_data['exp_avg_sq_shard']

                if 'params_shard' in fragment_data:
                    bucket.params_shard[slice(*fragment.shard_range)] = fragment_data['params_shard']
                    assert 'param_remainders_shard' not in fragment_data, fragment_data.keys()
                else:
                    assert (
                        bucket.params_shard is None
                    ), f'bucket.params_shard should be None, got: {bucket.params_shard}'
                    assert 'param_remainders_shard' in fragment_data, fragment_data.keys()

                if 'param_remainders_shard' in fragment_data:
                    _, rem = split_fp32(fragment_data['param_remainders_shard'])
                    bucket.param_remainders_shard[slice(*fragment.shard_range)] = rem

        # `fragments_local_data` was needed only to separate raw fragments data
        del optim_state['fragments_local_data']

        self.update_fp32_hyperparameters(state_dict)
        super().load_state_dict(state_dict)

        # We can't send bucket tensors to cuda earlier because `super()` copies all tensors (would OOM)
        for bucket in self.state['buckets']:
            for field in ('exp_avg_shard', 'exp_avg_sq_shard', 'params_shard', 'param_remainders_shard'):
                field_tensor = getattr(bucket, field)
                setattr(bucket, field, field_tensor.cuda() if field_tensor is not None else None)

    def optim_state_to_sharding_state(
        self, optim_state_dict: StateDict, id_to_sharded_param_map: Dict[int, ShardedTensor]
    ):
        """
        Wraps optimizer states with ShardedTensor based on model params ShardedTensors.

        Args:
            optim_state_dict: regular optimizer state dict
            id_to_sharded_param_map: a map from optimizer param ids to
                corresponding model param ShardedTensors.
                It will be used to create ShardedTensors for optimizer states.
        """
        optim_state = optim_state_dict['state']

        # For each model param we extract relevant data from the buckets,
        # wrap with ShardedTensor and store in `fragments_local_data`
        fragments_local_data = {}
        for param_id, param_state in optim_state.items():
            if not isinstance(param_id, int):
                continue
            fragments_local_data[param_id] = {
                fragment_id: self.make_sharded_fragment_data(
                    id_to_sharded_param_map[param_id], param_id, fragment, optim_state['buckets'][fragment.bucket_id]
                )
                for fragment_id, fragment in enumerate(param_state['fragments'])
                if fragment.in_local_shard
            }
        # When loading from checkpoint, we only need raw data from
        # `fragments_local_data`. All fragments and buckets metadata is taken
        # from the existing state (LocalNonpersitentObject).
        new_optim_state = {k: LocalNonpersitentObject(v) for k, v in optim_state.items() if isinstance(k, int)}
        new_optim_state['fragments_local_data'] = fragments_local_data
        new_optim_state['buckets'] = LocalNonpersitentObject(optim_state['buckets'])

        optim_state_dict['state'] = new_optim_state
        optim_state_dict['param_groups'] = deepcopy(optim_state_dict['param_groups'])
        for group in optim_state_dict['param_groups']:
            group['params'] = LocalNonpersitentObject(group['params'])
            # Step is saved to param_group for compatibility with regular optimizer
            group['step'] = optim_state['step']

        return optim_state_dict

    def make_sharded_fragment_data(
        self,
        model_param: ShardedTensor,
        param_id: int,
        fragment: DistributedFusedAdam.ParameterFragment,
        bucket: DistributedFusedAdam.StateBucket,
    ) -> Dict[str, ShardedTensor]:
        """
        Build a ShardedTensor for a given fragment.

        For sharding scheme explanation check: https://github.com/NVIDIA/Megatron-LM/blob/main/docs/distrib_optimizer.md#sharding-scheme
        DistributedFusedAdam.ParameterFragment attributes vs linked doc correspondence:
        - shard_range ~ local_index
        - shard_param_range ~ param_index
        - shard_bucket_range ~ world_index (not relevant here)
        For more details see DistributedFusedAdam docs.

        For each optimizer state field (exp_avg_shard, ...), this method
        extracts relevant data for the given fragment from the bucket
        (with `[slice(*fragment.shard_range)]`) and wraps with a ShardedTensor
        with similar attributes to the given `model_param` one.

        Args:
            model_param: model parameter corresponding to the given `param_id`
            param_id: optimizer param id
            fragment: one of the fragments for given `param_id`
            bucket: a bucket that contains data for the given fragment
        """
        fragment_local_data = {}
        assert param_id == fragment.param_id
        prefix = f'optimizer.state'
        simple_mapping_fields = {
            'exp_avg_shard': 'exp_avg',
            'exp_avg_sq_shard': 'exp_avg_sq',
            'params_shard': 'fp32_from_fp16',
            'param_remainders_shard': 'fp32_from_fp16',
        }

        for field, field_key in simple_mapping_fields.items():
            field_val = getattr(bucket, field)
            if field_val is None:
                continue
            optim_param = field_val[slice(*fragment.shard_range)]
            if field == 'param_remainders_shard':
                # Construct fp32 tensor from model BF16 params and INT16 remainders
                optim_param = merge_fp32(model_param.data.view(-1)[slice(*fragment.shard_param_range)], optim_param)
            assert len(model_param.replica_id) == 3, f'Expected replica_id format (TP, PP, DP), got: {replica_id}'
            replica_id = (*model_param.replica_id[:2], 0)
            fragment_local_data[field] = replace(
                model_param,
                key=f'{prefix}.{field_key}.{model_param.key}',
                data=optim_param,
                dtype=optim_param.dtype,
                flattened_range=slice(*fragment.shard_param_range),
                replica_id=replica_id,
            )

        return fragment_local_data

    def add_fp32_sharding_state(self, optim_state_dict, model_sharded_state_dict=None):
        """ Build sharded state dict for the params of FP32 optimizer. """
        if getattr(self, '_fp32_optim', None) is None:
            return
        if not isinstance(self._fp32_optim, torch.optim.AdamW):
            raise NotImplementedError(f'FP32 Optimizer of type {type(self._fp32_optim)} not supported')

        adam_init_state(self._fp32_optim)
        fp32_state_dict = self._fp32_optim.state_dict()  # recompute after state init

        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict, self._fp32_optim_main_params.keys()
        )

        def get_safe(param_id):
            try:
                return id_to_sharded_param_map[param_id]
            except KeyError as e:
                breakpoint()
                raise ValueError(f'Param id {param_id} does not match any model sharded param') from e

        # FP32 model params
        optim_state_dict['fp32_optim_fp32_params'] = [
            make_sharded_optimizer_tensor(get_safe(param_id), fp32_param, prefix=f'optimizer.state.fp32_from_fp16')
            for param_id, fp32_param in enumerate(optim_state_dict['fp32_optim_fp32_params'])
        ]

        # FP32 Optimizer state
        optim_state_to_sharding_state(fp32_state_dict, id_to_sharded_param_map, exclude_keys=('step',))

        # Since this is a wrapped optimizer, we don't want to store hyperparameters
        # but they must be updated with `update_fp32_hyperparameters` before calling `load_state_dict`
        for group_idx in range(len(fp32_state_dict['param_groups'])):
            # unwrap LocalNonpersitentObject from 'params' ...
            fp32_state_dict['param_groups'][group_idx]['params'] = fp32_state_dict['param_groups'][group_idx][
                'params'
            ].obj
            # ... and apply it to the whole group
            fp32_state_dict['param_groups'][group_idx] = LocalNonpersitentObject(
                fp32_state_dict['param_groups'][group_idx]
            )

        optim_state_dict['fp32_optim'] = fp32_state_dict

    def update_fp32_hyperparameters(self, state_dict):
        """ Copy relevant optimizer hyperparameters and step from main optimizer to FP32 one. """
        if 'fp32_optim' not in state_dict:
            return
        for main_group, fp32_group in zip(state_dict['param_groups'], state_dict['fp32_optim']['param_groups']):
            for k, v in main_group.items():
                if k in fp32_group and k != 'params' and fp32_group[k] != v:
                    logger.info(f'Replacing FP32 optimizer hparam {k} with {v} (previous value: {fp32_group[k]})')
                    fp32_group[k] = v
        # Copt step info
        step = state_dict['state']['step']
        for _, param_state in state_dict['fp32_optim']['state'].items():
            param_state['step'] = step


def adam_init_state(opt):
    for group in opt.param_groups:
        for p in group['params']:
            if len(opt.state[p]) == 0:
                opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)


def split_fp32(x):
    assert x.dtype is torch.float, x.dtype
    x = x.clone().detach()
    rem_bf16 = x.unsqueeze(-1).view(torch.int16)
    rem = rem_bf16[..., 0]
    bf16 = rem_bf16[..., 1]
    assert x.shape == rem.shape == bf16.shape, (x.shape, rem.shape, bf16.shape)
    # Round up BF16
    bf16 += torch.where(rem < 0, 1, 0)
    return bf16.view(torch.bfloat16), rem


def merge_fp32(bf16, rem):
    assert bf16.dtype is torch.bfloat16, bf16.dtype
    assert rem.dtype is torch.int16, rem.dtype
    # Round down BF16
    bf16 = bf16.clone().detach()
    bf16 -= torch.where(rem < 0, 1, 0)

    rem_bf16 = torch.stack((rem, bf16.view(torch.int16)), dim=-1)
    x = rem_bf16.view(torch.float32).squeeze(-1)
    assert x.shape == rem.shape == bf16.shape, (x.shape, rem.shape, bf16.shape)
    return x
