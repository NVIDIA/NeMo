# mypy: allow-untyped-defs
import copy
import io
import weakref
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist

if dist.is_available() or TYPE_CHECKING:
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._tensor import DTensor


def _identity_func(
    obj: torch.Tensor,
    pg: Optional[dist.ProcessGroup],
    device: Optional[torch.device],
    companion_obj: Any,
) -> torch.Tensor:
    """ noop """
    return obj


class CompanionMismatch(Exception): ...


def _iterate_state_dict(
    iter_object: Any,
    sharded_tensor_func: Callable,
    dtensor_func: Callable,
    tensor_func: Callable,
    *,
    pg: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
    cpu_offload: bool = False,
    companion_obj: Any = None,
    ranks_only: Tuple[int, ...] = (),
    type_check: bool = True,
    non_blocking: bool = True,
) -> Dict[str, Any]:
    """Iterate through the state dict, applying the given functions to each tensor type.

    Args:
        iter_object (Any): the target state_dict.
        sharded_tensor_func (Callable): the function to apply to ShardedTensor
        dtensor_func (Callable): the function to apply to DTensor
        tensor_func (Callable): the function to apply to Tensor
        pg (Optional[dist.ProcessGroup]): process group passed to tensor functions
        device (Optional[torch.device]): device passed to tensor functions
        cpu_offload (bool): whether to offload the tensors to CPU memory. This option is ignored
            if a companion_obj is supplied.
        companion_obj (Any): A companion object to the state dict. If this object
            is supplied, we attempt to copy the tensor to the companion object.
        ranks_only (Tuple[int, ...]): if this tuple is empty, all ranks will
            have the same state_dicts. Otherwise only ranks that in ``ranks_only``
            have the same state_dicts. Other ranks will get empty state_dicts.
        type_check (bool): check if the instance data type is a supported type
            that can be saved by DCP.  The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.
        non_blocking (bool): whether to use non-blocking copy when copying to the companion object.
    """
    # TODO: should we use pytree?
    cpu_device = torch.device("cpu")
    if isinstance(iter_object, ShardedTensor):
        ret = sharded_tensor_func(iter_object, pg, device, companion_obj)
    elif isinstance(iter_object, DTensor):
        ret = dtensor_func(iter_object, pg, device, companion_obj)
    elif isinstance(iter_object, torch.Tensor):
        ret = tensor_func(iter_object, pg, device, companion_obj)
    elif isinstance(iter_object, (int, float, str, bytes, io.BytesIO)) or iter_object is None:
        ret = iter_object
    elif isinstance(iter_object, dict):
        if companion_obj is not None and (
            not isinstance(companion_obj, dict) or set(companion_obj.keys()) != set(iter_object.keys())
        ):
            msg = "" if isinstance(companion_obj, dict) else f"{set(companion_obj.keys())=} {set(iter_object.keys())=}"
            raise CompanionMismatch(msg)

        ret = {
            key: _iterate_state_dict(
                value,
                sharded_tensor_func,
                dtensor_func,
                tensor_func,
                pg=pg,
                device=device,
                cpu_offload=cpu_offload,
                companion_obj=companion_obj[key] if companion_obj is not None else None,
                ranks_only=ranks_only,
                type_check=type_check,
                non_blocking=non_blocking,
            )
            for key, value in iter_object.items()
        }
    elif isinstance(iter_object, (list, tuple)):
        if companion_obj is not None and (
            not isinstance(companion_obj, (list, tuple)) or len(companion_obj) != len(iter_object)
        ):
            raise CompanionMismatch

        ret = [
            _iterate_state_dict(
                v,
                sharded_tensor_func,
                dtensor_func,
                tensor_func,
                pg=pg,
                device=device,
                cpu_offload=cpu_offload,
                companion_obj=companion_obj[idx] if companion_obj is not None else None,
                ranks_only=ranks_only,
                type_check=type_check,
                non_blocking=non_blocking,
            )
            for idx, v in enumerate(iter_object)
        ]
        if isinstance(iter_object, tuple):
            ret = tuple(ret)
    # Logic for Megatron ShardedTensor and ShardedObject abstractions
    elif any(_ in str(type(iter_object)) for _ in ["mapping.ShardedTensor", "mapping.ShardedObject"]):
        if hasattr(iter_object, "data"):
            if isinstance(iter_object.data, torch.Tensor):
                ret_data = tensor_func(iter_object.data, pg, device, companion_obj)
            else:
                ret_data = copy.deepcopy(iter_object.data)
            ret = copy.copy(iter_object)
            ret_dict = {k: copy.deepcopy(v) for k, v in iter_object.__dict__.items() if k != "data"}
            ret.__dict__.update(ret_dict)
            ret.data = ret_data
        else:
            ret = copy.deepcopy(iter_object)
    elif not type_check:
        ret = copy.deepcopy(iter_object)
    else:
        raise ValueError(f"Unexpected value type {type(iter_object)}")

    if not ranks_only or dist.get_rank(pg) in ranks_only:
        if isinstance(ret, torch.Tensor):
            if cpu_offload and companion_obj is None:
                ret = ret.to(cpu_device)

            if companion_obj is not None:
                # TODO: support DTensor
                companion_obj.copy_(ret, non_blocking=non_blocking)
                ret = companion_obj

        # Logic for Megatron ShardedTensor and ShardedObject abstractions
        elif any(_ in str(type(iter_object)) for _ in ["mapping.ShardedTensor", "mapping.ShardedObject"]):
            if cpu_offload and companion_obj is None:
                ret.data = ret.data.detach().to(cpu_device)

            if companion_obj is not None:
                # companion_obj_dict = {k: copy.deepcopy(v) for k, v in ret.__dict__.items() if k != "data"}
                companion_obj.__dict__.update(ret_dict)
                if isinstance(companion_obj.data, torch.Tensor):
                    if ret.data.requires_grad:
                        companion_obj.data.copy_(ret.data.detach(), non_blocking=non_blocking)
                    elif not companion_obj.data.requires_grad and not ret.data.requires_grad:
                        companion_obj.data.copy_(ret.data, non_blocking=non_blocking)
                    else:
                        raise ValueError("Incompatible requires_grad values")
                else:
                    companion_obj.data = copy.deepcopy(ret.data)
                # TODO: support DTensor
                ret = companion_obj
    else:
        ret = {} if isinstance(ret, dict) else None

    return ret


def _copy_state_dict(
    state_dict: Dict[str, Any],
    copy_state_dict: Dict[str, Any],
    non_blocking: bool = False,
    type_check: bool = True,
) -> Dict[str, Any]:
    """
    Copies all tensors in a given state dict into a different state_dict with the
    same structure. Additionally, a copied state dict with the same value references
    is returned. Editing the keys on this state dict will not affect the
    passed in copy_state_dict (but the value references are the same).

    .. warning::
        It is expected by this function that state_dict and copy_state_dict share
        the same structure and data types.

    .. warning::
        The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Args:
        state_dict (Dict[str, Any]): the target state_dict.
        copy_state_dict (Dict[str, Any]):
            The state dict we are copying into. This state_dict must have exactly
             the same structure as the source `state_dict`.
        non_blocking: (bool): Whether copy ops should be performed asynchronously
        type_check (bool): check if the instance data type is a supported type
            that can be saved by DCP. The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Returns:
        State Dict copy
    """

    return _iterate_state_dict(
        state_dict,
        _identity_func,
        _identity_func,
        _identity_func,
        pg=None,
        device=None,
        cpu_offload=False,
        ranks_only=(),
        companion_obj=copy_state_dict,
        type_check=type_check,
        non_blocking=non_blocking,
    )


def _create_cpu_state_dict(
    state_dict: Dict[str, Any], pin_memory: bool = False, share_memory: bool = False
) -> Dict[str, Any]:
    """
    Given a state_dict, create another state_dict with the same structure and elements.
    However, all tensors in the returned state_dict are new tensors on CPU. These
    tensors can be placed on pin_memory or share_memory based on the provided arguments.

    .. warning::
        Setting both `pin_memory` and `share_memory` to True significantly increases the
        latency of this method because of the nuances which require us to register memory
        as pinned directly as opposed to relying on the pin_memory cache allocator. This
        option should only be used for long lived tensors which are required to be shared.
        This is not the case as long as at least one of `pin_memory` or `share_memory` is
         set to False.

    """

    def tensor_func(
        obj: torch.Tensor,
        pg: Optional[dist.ProcessGroup],
        device: Optional[torch.device],
        _: Any,
    ) -> torch.Tensor:
        if len(obj.size()) == 0:
            return torch.tensor(0, dtype=obj.dtype)

        if share_memory:
            t = torch.empty(*tuple(obj.size()), dtype=obj.dtype)
            t = t.share_memory_()
            if pin_memory:

                def unpin_memory(t):
                    succ = int(torch.cuda.cudart().cudaHostUnregister(t.data_ptr()))
                    assert succ == 0, f"Unpinning shared memory failed with error-code: {succ}"

                weakref.finalize(t, unpin_memory, t)
                succ = int(
                    torch.cuda.cudart().cudaHostRegister(
                        t.data_ptr(),
                        t.numel() * t.element_size(),
                        1,  # lines up with 'cudaHostRegisterPortable'
                    )
                )
                assert succ == 0, f"Pinning shared memory failed with error-code: {succ}"
            return t
        elif pin_memory:
            return torch.empty(*tuple(obj.size()), dtype=obj.dtype).pin_memory()
        else:
            return torch.empty(*tuple(obj.size()), dtype=obj.dtype)

    ret = _iterate_state_dict(
        state_dict,
        _identity_func,
        _identity_func,
        tensor_func,
        pg=None,
        device=None,
        cpu_offload=False,
        ranks_only=(),
        type_check=False,
    )
    return ret


def _offload_state_dict_to_cpu(
    state_dict: Dict[str, Any],
    *,
    ranks_only: Tuple[int, ...] = (),
    type_check: bool = True,
) -> Dict[str, Any]:
    """
    Given a state_dict, this API offload all the tensors to CPU memory.

    Args:
        state_dict (Dict[str, Any]): the target state_dict.
        pg (Optional[dist.ProcessGroup]): the process group that is used to
            gather ShardedTensor. Note that gathering a DTensor will use
            the DeviceMesh. So this argument will be ignored when gathering a
            DTensor.
        ranks_only: (Tuple[int, ...]): if this tuple is empty, all ranks will
            have the same state_dicts. Otherwise only ranks that in ``ranks_only``
            have the same state_dicts. Other ranks will get empty state_dicts.
        type_check: (bool): check if the instance data type is a supported type
            that can be saved by DCP.  The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Returns:
        The gathered state dictionary.
    """

    ret = _iterate_state_dict(
        state_dict,
        _identity_func,
        _identity_func,
        _identity_func,
        pg=None,
        device=None,
        cpu_offload=True,
        ranks_only=ranks_only,
        type_check=type_check,
    )
    return ret
