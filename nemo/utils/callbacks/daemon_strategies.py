import os
import queue
import threading
from itertools import chain
from logging import getLogger
from pathlib import Path
from time import time
from typing import Callable, List, Optional, Tuple, Union

import torch
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest
from megatron.core.dist_checkpointing.strategies.filesystem_async import (
    FileSystemWriterAsync,
    WriteBucket,
    _disable_gc,
    _process_memory,
    _split_by_separation_hint,
    _split_by_size_and_type,
)
from megatron.core.dist_checkpointing.strategies.state_dict_saver import save_state_dict_async_plan
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreSavePlanner,
    TorchDistSaveShardedStrategy,
    _replace_state_dict_keys_with_sharded_keys,
    mcore_to_pyt_state_dict,
)
from torch import multiprocessing as mp
from torch.distributed.checkpoint.filesystem import DEFAULT_SUFFIX, _StoragePrefix, _write_item
from torch.distributed.checkpoint.planner import SavePlan, SavePlanner, WriteItemType
from torch.distributed.checkpoint.storage import WriteResult

logger = getLogger(__name__)

class DaemonTorchDistSaveShardedStrategy(TorchDistSaveShardedStrategy):
    """ An asynchronous strategy for saving PyTorch sharded state dictionaries in a dist env.

    This class extends TorchDistSaveShardedStrategy and provides methods to:
    1. Translate MCore sharded state dictionaries to PyTorch's native format.
    2. Save them asynchronously using a daemon-based file system writer.
    3. Optionally cache and reuse checkpoint structure (plans and metadata)
        for faster subsequent saves.

    Args:
        backend (str): The distributed backend in use.
        version (int): Checkpoint version number.
        keep_only_main_replica (bool, optional): If True, deduplicates and keeps only the main
            replica's tensors before saving. Defaults to True.
        thread_count (int, optional): Number of threads the daemon-based writer should use.
            Defaults to 2.
        cached_metadata (bool, optional): If True, attempts to cache and reuse pre-computed
            remote metadata. Defaults to False.
        separation_hint (str, optional): Logical grouping hint (e.g., "metadata" or "weights") that
            the daemon-based writer can use for separating data in the file system. Defaults to None.

    Attributes:
        cached_central_plan (Optional[Any]): Cached plan for the central process group if
            caching is enabled.
        cached_local_plan (Optional[Any]): Cached plan for local process group members if
            caching is enabled.
        validated_cache_reuse (bool): Whether the cached plan is validated for subsequent reuse.
        cached_global_metadata (Optional[Any]): If caching is enabled, holds the metadata
            required for globally recognized checkpoint details.

    Methods:
        save(sharded_state_dict, checkpoint_dir):
            Synchronous wrapper around the asynchronous save. Calls async_save(...) and then
            executes the returned finalize functions immediately, effectively blocking until
            completion.

        async_save(sharded_state_dict, checkpoint_dir) -> AsyncRequest:
            Translates the MCore sharded state dictionary to a native PyTorch state dictionary,
            initiates an asynchronous save routine with a daemon-based file system writer,
            and caches checkpoint structures if enabled. Returns an AsyncRequest containing
            the necessary asynchronous callbacks for finalization.
    """
    def __init__(
        self,
        backend: str,
        version: int,
        keep_only_main_replica: bool = True,
        thread_count: int = 2,
        cached_metadata: bool = False,
        separation_hint: str = None,
    ):
        super().__init__(
            backend,
            version,
            keep_only_main_replica,
            thread_count,
            cached_metadata,
            separation_hint,
        )

    def save(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        """Each async strategy can be trivially used as a sync strategy."""
        async_request = self.async_save(sharded_state_dict, checkpoint_dir)
        async_request.async_fn(*async_request.async_fn_args)
        for finalize_fn in async_request.finalize_fns:
            finalize_fn()

    def async_save(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> AsyncRequest:
        """Translates MCore ShardedTensors to PyT ShardedTensors & saves in PyT Distributed format.

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict to save
            checkpoint_dir (Path): checkpoint directory

        Returns: None
        """
        # Translate the state dict
        (sharded_state_dict, flat_mapping, rename_mapping) = _replace_state_dict_keys_with_sharded_keys(
            sharded_state_dict, self.keep_only_main_replica
        )
        pyt_state_dict = mcore_to_pyt_state_dict(sharded_state_dict, False)
        # Use PyT saving mechanism
        writer = DaemonFileSystemWriterAsync(
            checkpoint_dir,
            separation_hint=self.separation_hint,
            thread_count=self.thread_count,
        )
        # This should be set differently if we run in a smaller process group than the default
        coordinator = 0
        # Try twice to validate the generated `central_plan` is the same across iterations
        # If so, reuse `cached_central_plan` and `cached_global_metadata`
        # From the 3rd iteration, `save_state_dict_async_plan` will not generate `global_metadata`
        # (return None) so `self.cached_global_metadata` is reused
        args_cached_plans = None
        if self.use_cached_ckpt_structure:
            args_cached_plans = (
                self.cached_central_plan,
                self.cached_local_plan,
                self.validated_cache_reuse,
            )

        (
            save_state_dict_ret,
            self.cached_central_plan,
            self.cached_local_plan,
            self.validated_cache_reuse,
        ) = save_state_dict_async_plan(
            pyt_state_dict,
            writer,
            None,
            coordinator,
            planner=MCoreSavePlanner(dedup_replicated_tensors=not self.keep_only_main_replica),
            cached_ckpt_structure=args_cached_plans,
        )
        rank = torch.distributed.get_rank()
        if self.use_cached_ckpt_structure:
            if self.validated_cache_reuse:
                logger.debug(f"rank: {rank}, cache validated")
                if save_state_dict_ret[1]:  # when global_metadata is not cached
                    self.cached_global_metadata = save_state_dict_ret[1]  # Cache Metadata
                # Only Coordinator rank holds cached global_metadata
                # (None is returned for global_metadata)
                elif coordinator == rank:
                    logger.debug(f"rank: {rank}, reuse metadata, {save_state_dict_ret[1]}")
                    save_state_dict_ret = list(save_state_dict_ret)
                    save_state_dict_ret[1] = self.cached_global_metadata

        return self._get_save_and_finalize_callbacks(writer, save_state_dict_ret)


class DaemonFileSystemWriterAsync(FileSystemWriterAsync):
    """ DaemonFileSystemWriterAsync
        TODO
    """
    def __init__(self, *args, separation_hint: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.separation_hint = separation_hint
        self.write_results_or_exc_bucket = []

    def prepare_write_data(self, plan: SavePlan, planner: SavePlanner) -> None:
        """
        First stage of async saving. Copy data to CPU and plan the local saving.

        Args:
            plan (SavePlan): save plan generated by the PyT Distributed compatible planner
            planner (SavePlanner): save planner used to resolve the bytes and tensor data

        Returns: None, but stores the save plan in `self.write_buckets`
        """
        storage_plan: _StoragePrefix = plan.storage_data
        start = time()
        logger.debug(f"thread_count: {self.thread_count}, time: {start}")
        if self.separation_hint:
            assert self.thread_count > 1, "thread_count must be at least 2 if separation_hint is provided"
        bins = self.thread_count // 2 if self.separation_hint is not None else self.thread_count
        item_buckets = _split_by_size_and_type(bins, plan.items, self.separation_hint)
        logger.debug(f"bucket_prep, time: {time() - start}")

        start = time()
        # move tensors from GPU to CPU before starting async writing
        # We do D2H synchronously for now
        file_count = 0

        def gen_file(prefix=""):
            nonlocal file_count
            file_name = f"{prefix}{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}"
            file_count += 1
            return file_name

        # Prepare bytes / tensor data in each bucket, which will be assigned to each writer process
        self.write_buckets = []
        for group_name, group_buckets in _split_by_separation_hint(item_buckets, self.separation_hint).items():
            for bucket in group_buckets:
                bytes_data = [
                    (item, planner.resolve_data(item)) for item in bucket if item.type == WriteItemType.BYTE_IO
                ]
                tensor_data = [
                    (
                        item,
                        planner.resolve_data(item).detach().to("cpu", non_blocking=True),
                    )
                    for item in bucket
                    if item.type != WriteItemType.BYTE_IO
                ]
                if len(bytes_data) > 0 or len(tensor_data) > 0:
                    file_name = gen_file(prefix=group_name)
                    self.write_buckets.append((self.path / file_name, file_name, (bytes_data, tensor_data)))

        end = time()
        logger.debug(f"D2H and push, time: {end - start}")

    def get_save_function_and_args(self) -> Tuple[Optional[Callable], Tuple]:
        """
        Get function that saves the data to storage along with its arguments.
        Allows the external caller to apply the save function synchronously or asynchronously.

        Returns: None (if there is nothing to write on this rank) or a tuple of:
            - the function that saves the data
            - arguments to that function
        """
        if not self.write_buckets:
            return None, ()
        return (
            self.write_preloaded_data_subproc,
            (self.write_buckets, self.write_results_or_exc_bucket),
        )

    @staticmethod
    @_disable_gc()
    def write_preloaded_data_subproc(
        write_buckets: List[WriteBucket],
        write_results_or_exc_bucket: List[Union[dict, Exception]],
    ) -> None:
        """
        Performs saving data to storage with multiple processes.

        Starts predefined number of processes and uses 2 queues to make sure the results
        are complete:
        - local_results_queue - to send the actual results
        - count_queue - small queue to mark worker as completed

        Using just one queue disallowed proper exception handling.

        This method is meant to be run in a forked subprocess.
        Triggering GC during execution leads to CUDA errors
        (cleaning up tensors owned by the parent process).
        To prevent this, we disable the GC explicitly for this function with _disable_gc.

        Args:
            write_buckets (List[WriteBucket]): write plan
        Returns: None
        """
        w_start = time()
        write_results_or_exc: Union[dict, Exception] = dict()
        ctx = mp.get_context("fork")
        local_results_queue = ctx.Queue()
        threads = []
        for i, write_bucket in enumerate(write_buckets):
            try:
                threads.append(
                    threading.Thread(
                        target=DaemonFileSystemWriterAsync.write_preloaded_data,
                        args=(i, write_bucket, local_results_queue, True),
                    )
                )
            except Exception as e:
                err_msg = f"An error is caught while a proc {i} is created, error: {e}"
                logger.error(err_msg)
                write_results_or_exc = RuntimeError(err_msg)

        if not isinstance(write_results_or_exc, Exception):
            for t in threads:
                t.start()

            logger.debug("FileSystemWriterAsync: collecting worker results...")

            for t in threads:
                t.join()

            # At this point, all workers completed, so the queue should have exactly
            # `len(write_buckets)` items
            for proc_idx in range(len(write_buckets)):
                try:
                    local_proc_idx, local_results_or_exc = local_results_queue.get()
                except queue.Empty:
                    write_results_or_exc = RuntimeError(
                        f"Unexpected empty `local_results_queue`" f" (got only {proc_idx}/{len(write_buckets)} items)"
                    )
                    break
                else:
                    if isinstance(local_results_or_exc, Exception):
                        err_msg = f"Local process {local_proc_idx} encountered" f" an error: {local_results_or_exc}"
                        logger.error(err_msg)
                        write_results_or_exc = local_results_or_exc
                        break
                    else:
                        assert isinstance(local_results_or_exc, list), type(local_results_or_exc)
                        write_results_or_exc[local_proc_idx] = local_results_or_exc

            logger.debug("DaemonFileSystemWriterAsync: collected worker results successfully")

        w_end = time()
        logger.debug(f"{w_end}, rank: {torch.distributed.get_rank()}," f" write(sync,parallel): {w_end - w_start}")

        write_results_or_exc_bucket.append(write_results_or_exc)

    @staticmethod
    @_disable_gc()
    def write_preloaded_data(
        local_proc_idx: int,
        write_bucket: WriteBucket,
        results_queue: mp.Queue,
        use_fsync: bool,
    ) -> None:
        """
        Performs actual data saving to storage.

        Args:
            local_proc_idx (int): index of a local process that performs writing
            write_bucket (WriteBucket): data to write to storage
            results_queue (mp.Queue): queue to return the write results
                to the proxy checkpoint process.
            use_fsync (bool): if True, calls os.fsync at the end of saving

        Returns: None, the write result are put into the `queue`
        """
        mem_before = _process_memory()

        local_results = []
        try:
            file_name, storage_key, (bytes_data, tensor_data) = write_bucket
            with open(file_name, "wb") as stream:
                for write_item, data in bytes_data:
                    local_results.append(_write_item(stream, data, write_item, storage_key))

                for write_item, tensor in tensor_data:
                    assert tensor.is_cpu
                    local_results.append(_write_item(stream, tensor, write_item, storage_key))

                if use_fsync:
                    os.fsync(stream.fileno())
            local_output = (local_proc_idx, local_results)
        except Exception as e:
            local_output = (local_proc_idx, e)

        results_queue.put(local_output)

        mem_after = _process_memory()
        logger.debug(
            f"{local_proc_idx} consumed: {mem_after - mem_before}," f" before: {mem_before}, after: {mem_after}"
        )

    def retrieve_write_results(self) -> List[WriteResult]:
        """
        Turn the latest dict including write results from `self.write_results_or_exc`
            into a single results lists. Includes error check.

        Returns (List[WriteResult]): the list of write results
            from all local processes performing the save.

        """
        write_results_or_exc = self.write_results_or_exc_bucket.pop()

        if isinstance(write_results_or_exc, Exception):
            raise RuntimeError(f"Worker failure: {write_results_or_exc}") from write_results_or_exc
        write_results: dict = write_results_or_exc
        if len(write_results) != len(self.write_buckets):
            raise RuntimeError(
                f"Incomplete worker results (expected {len(self.write_buckets)},"
                f" got {len(write_results)}. This probably indicates a worker failure."
            )
        return list(chain.from_iterable(write_results.values()))
