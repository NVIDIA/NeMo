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
# limitations under the License.

import signal
from typing import Any, List, Optional

import torch
import torch.distributed

from nemo.tron.utils.common_utils import get_world_size_safe, print_rank_0


def get_device(local_rank: Optional[int] = None) -> torch.device:
    """Get the appropriate torch device based on the distributed backend.

    Args:
        local_rank: The local rank, used to specify the CUDA device index for NCCL.
                    If None, uses the default CUDA device.

    Returns:
        The torch.device ('cuda' for NCCL, 'cpu' for Gloo).

    Raises:
        RuntimeError: If the distributed backend is neither 'nccl' nor 'gloo'.
    """
    backend = torch.distributed.get_backend()
    if backend == "nccl":
        if local_rank is None:
            device = torch.device("cuda")
        else:
            device = torch.device(f"cuda:{local_rank}")
    elif backend == "gloo":
        device = torch.device("cpu")
    else:
        raise RuntimeError
    return device


def all_gather_item(
    item: Any,
    dtype: torch.dtype,
    group: Optional[torch.distributed.ProcessGroup] = None,
    async_op: bool = False,
    local_rank: Optional[int] = None,
) -> List[Any]:
    """Perform an all_gather operation on a single Python object.

    Converts the item to a tensor, performs all_gather, and converts back to a list
    of Python objects from all ranks.

    Args:
        item (Any): The Python object to gather.
        dtype (torch.dtype): The torch dtype to use for the intermediate tensor.
        group (Optional[torch.distributed.ProcessGroup]): The process group to gather within (defaults to the default group).
        async_op (bool): Whether the operation should be asynchronous.
        local_rank (Optional[int]): The local rank to determine the device.

    Returns:
        List[Any]: A list containing the gathered items (of type Any) from all ranks in the group.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return [item]

    device = get_device(local_rank)

    if group is not None:
        group_size = group.size()
    else:
        group_size = get_world_size_safe()

    tensor = torch.tensor([item], device=device, dtype=dtype)
    output_tensors = [torch.zeros(1, dtype=tensor.dtype, device=tensor.device) for _ in range(group_size)]
    torch.distributed.all_gather(output_tensors, tensor, group, async_op)
    output = [elem.item() for elem in output_tensors]
    return output


class DistributedSignalHandler:
    """Context manager to handle signals gracefully in a distributed setting.

    Installs a signal handler upon entering the context that sets a flag
    when the specified signal is received. The `signals_received` method
    can be used to check if any rank received the signal (using all_gather).
    The original signal handler is restored upon exiting the context.

    Args:
        sig: The signal number to handle (e.g., signal.SIGTERM).
             Defaults to signal.SIGTERM.
    """

    def __init__(self, sig: int = signal.SIGTERM) -> None:
        self.sig = sig
        self._signal_received = False
        self.released = False
        self.original_handler = None

    def signals_received(self) -> List[bool]:
        """Check if any rank in the default group received the signal.

        Uses all_gather to collect the signal status from all ranks.

        Returns:
            A list of booleans, where each element indicates if the
            corresponding rank received the signal.
        """
        all_received = all_gather_item(self._signal_received, dtype=torch.int32)
        return all_received

    def __enter__(self) -> "DistributedSignalHandler":
        self._signal_received = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)

        def handler(signum: int, frame: Optional[Any]) -> None:
            print_rank_0(f"Received signal {signum}, initiating graceful stop")
            self._signal_received = True

        signal.signal(self.sig, handler)
        print_rank_0(f"Signal handler installed for {self.sig}")

        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        """Release the signal handler and restore the original handler."""
        self.release()

    def release(self) -> bool:
        """Restore the original signal handler.

        Returns:
            True if the handler was released, False if it was already released.
        """
        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True
