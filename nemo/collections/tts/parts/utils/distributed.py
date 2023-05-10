from typing import Iterable

import torch


def _is_distributed():
    return torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1


def _is_complex_or_float(tensor):
    return torch.is_floating_point(tensor) or torch.is_complex(tensor)


def broadcast_tensors(tensors: Iterable[torch.Tensor], src: int = 0):
    """
    Broadcast the tensors from the given parameters to all workers.
    This can be used to ensure that all workers have the same model to start with.
    """
    if not _is_distributed():
        return

    tensors = [tensor for tensor in tensors if _is_complex_or_float(tensor)]
    handles = []
    for tensor in tensors:
        handle = torch.distributed.broadcast(tensor.data, src=src, async_op=True)
        handles.append(handle)
    for handle in handles:
        handle.wait()
