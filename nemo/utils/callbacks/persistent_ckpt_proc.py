import logging
from logging import getLogger
from typing import Any, Dict, Optional

from lightning.fabric.utilities.types import _PATH
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue, debug_time

from nemo.uitls._ckpt_utils import TorchCompatiblePersistentAsyncCaller
from nemo.utils.callbacks.dist_ckpt_io import (
    AsyncCompatibleCheckpointIO,
    AsyncFinalizableCheckpointIO,
    _WrappingCheckpointIO,
)

logger = getLogger(__name__)


class PersistentCheckpointProcessIO(AsyncFinalizableCheckpointIO):
    def __init__(
        self,
        checkpoint_io: AsyncCompatibleCheckpointIO,
        profile_dir: Optional[str] = None,
    ) -> None:
        _WrappingCheckpointIO.__init__(self, checkpoint_io)
        self.async_calls_queue = AsyncCallsQueue(persistent=True)
        self.profile_dir = profile_dir

    @debug_time("PersistentCheckpointProcessIO.save_checkpoint")
    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
        """Executes async request returned from the underlying checkpoint_io asynchronously.

        Requires the underlying checkpoint_io.save_checkpoint to return an AsyncRequest.
        It is then applied with `self.async_calls_queue` asynchronously.

        Args:
            checkpoint (Dict[str, Any]): checkpoint to save. Passed to underlying
                checkpoint_io without modifications.
            path (_PATH): path to save the checkpoint. Passed to underlying
                checkpoint_io without modifications.
            storage_options (Any, optional): storage control modifiers. This class
                consumed the `finalize_fn` parameter (if any), which is expected to be
                a callback and is appended to async finalization functions.

        Applies underlying checkpoint_io finalize callback first, then the external one (postfix order).
        """

        external_finalize_fn = (storage_options or {}).pop("finalize_fn", None)
        assert isinstance(self.checkpoint_io, AsyncCompatibleCheckpointIO), type(self.checkpoint_io)

        if self.async_calls_queue.persistent_caller is None:
            self.async_calls_queue.persistent_caller = TorchCompatiblePersistentAsyncCaller(self.profile_dir)

        self.maybe_finalize_save_checkpoint(blocking=True)

        async_req = self.checkpoint_io.save_checkpoint(checkpoint, path, storage_options)
        if external_finalize_fn is not None:
            async_req.add_finalize_fn(external_finalize_fn)

        call_idx = self.async_calls_queue.schedule_async_request(async_req)

        logging.debug(f"Scheduled an async call #{call_idx}")
