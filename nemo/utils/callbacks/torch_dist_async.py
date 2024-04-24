from collections import deque
from logging import getLogger
from pathlib import Path
from time import time
from typing import Callable, List, NamedTuple, Optional, Tuple

import torch
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync
from megatron.core.dist_checkpointing.strategies.state_dict_saver import (
    save_state_dict_async_finalize,
    save_state_dict_async_plan,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreSavePlanner,
    TorchDistSaveShardedStrategy,
    _replace_state_dict_keys_with_sharded_keys,
    mcore_to_pyt_state_dict,
)
from torch import multiprocessing as mp

logger = getLogger(__name__)


class TorchDistAsyncSaveShardedStrategy(TorchDistSaveShardedStrategy):
    """Async save strategy for the PyT Distributed format. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.async_writer: Optional[DistributedAsyncCaller] = None
        self.is_async_save_active: bool = False

    def save(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        """ Translates MCore ShardedTensors to PyT ShardedTensors and saves in PyT Distributed format.

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict to save
            checkpoint_dir (Path): checkpoint directory

        Returns: None
        """
        # Translate the state dict
        (sharded_state_dict, flat_mapping, rename_mapping,) = _replace_state_dict_keys_with_sharded_keys(
            sharded_state_dict, self.keep_only_main_replica
        )
        pyt_state_dict = mcore_to_pyt_state_dict(sharded_state_dict, False)
        # Use PyT saving mechanism
        writer = FileSystemWriterAsync(checkpoint_dir, thread_count=self.thread_count)

        self.maybe_finalize_async_save(blocking=True)

        self.save_state_dict_ret = save_state_dict_async_plan(
            pyt_state_dict,
            writer,
            None,
            planner=MCoreSavePlanner(dedup_replicated_tensors=not self.keep_only_main_replica),
        )
        fun_args = writer.get_save_function_and_args()

        if self.async_writer is None:
            self.async_writer = DistributedAsyncCaller()
        if fun_args is not None:
            self.async_writer.schedule_async_call(*fun_args)
        self.is_async_save_active = True

    def maybe_finalize_async_save(self, blocking=False) -> bool:
        if not self.is_async_save_active:
            return False

        # We don't need a barrier, there is one in save_state_dict_async_finalize
        async_done = self.async_writer.is_current_async_call_done(blocking, post_barrier=False)
        if not async_done:
            return async_done

        self.do_finalize_async_save()
        return True

    def do_finalize_async_save(self) -> None:
        if self.save_state_dict_ret is None:
            raise CheckpointingException('finalize_async_save called, but no ckpt save in progress')

        # Pytorch Dist format requires metadata gathering in `post_async_save`
        save_state_dict_async_finalize(*self.save_state_dict_ret)
        self.is_async_save_active = False
        torch.distributed.barrier()


class DistributedAsyncCaller:
    def __init__(self):
        self.process = None
        self.start_time = None

    def schedule_async_call(
        self, async_fn: Callable, save_args: Tuple,
    ):
        """ Spawn a saving process"""
        torch.cuda.synchronize()
        ctx = mp.get_context('fork')
        self.start_time = time()
        self.process = ctx.Process(target=async_fn, args=save_args,)
        self.process.start()

    def close(self):
        if self.process:
            self.process.join()

    def is_current_async_call_done(self, blocking=False, post_barrier=True):
        """ Check if async save is finished.

        Returns True for the first call of this method after an async save is done
        (or in progress if blocking=True).

        # We assume all ranks have AsyncWriter.
        """
        # The following takes the same overhead as torch.distributed.barrier (single integer all-reduce)
        if self.process is not None:
            is_alive = int(self.process.is_alive())
        else:
            is_alive = 0
        ten = torch.tensor([is_alive], dtype=torch.int, device=torch.cuda.current_device())
        logger.debug(f"rank: {torch.distributed.get_rank()}, {ten}")
        torch.distributed.all_reduce(ten)
        if ten[0] > 0 and not blocking:
            return False
        else:
            if self.process is not None:
                logger.debug(f"rank: {torch.distributed.get_rank()}, joining self.process")
                self.process.join()
                self.process = None

                logger.debug(
                    f"DistributedAsyncCaller: Async process join finished after {time() - self.start_time:.2f}s from forking"
                )
                self.start_time = None

            # This ensures no race condition on `if self.process` during next is_current_async_call_done call
            if post_barrier:
                torch.distributed.barrier()
            return True


class _AsyncCall(NamedTuple):
    async_caller: DistributedAsyncCaller
    finalize_callback: Callable[[], None]


class AsyncCallsQueue:
    def __init__(self):
        self.async_calls = deque([])

    def schedule_async_call(self, async_fn: Callable, save_args: Tuple, finalize_fn: Callable):
        async_caller = DistributedAsyncCaller()
        async_caller.schedule_async_call(async_fn, save_args)
        self.async_calls.append(_AsyncCall(async_caller, finalize_fn))

    def maybe_finalize_async_calls(self, blocking=False) -> int:
        """ Finalizes all available calls. """
        num_calls_finalized = 0
        while self.async_calls:
            next_async_done = self.async_calls[0].is_current_async_call_done(blocking)
            if not next_async_done:
                break
            num_calls_finalized += 1
            _, finalize_fn = self.async_calls.popleft()
            finalize_fn()
        return num_calls_finalized
