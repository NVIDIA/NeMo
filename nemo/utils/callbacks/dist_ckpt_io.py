import shutil
from abc import abstractmethod
from contextlib import contextmanager
from time import time
from typing import Any, Dict, Optional

from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.types import _PATH
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.strategies import tensorstore

from nemo.utils import logging
from nemo.utils.callbacks.torch_dist_async import \
    TorchDistAsyncSaveShardedStrategy


class AsyncFinalizableCheckpointIO(CheckpointIO):
    @abstractmethod
    def maybe_finalize_save_checkpoint(self, blocking: bool = False):
        raise NotImplementedError


@contextmanager
def debug_time(name: str):
    start = time()
    try:
        yield
    finally:
        logging.debug(f'{name} took {time() - start:.3f}s')


class DistributedCheckpointIO(AsyncFinalizableCheckpointIO):
    """ CheckpointIO for a distributed checkpoint format.

    Args:
        save_ckpt_format (str): Distributed checkpoint format to use for checkpoint saving.
    """

    def __init__(
            self,
            save_ckpt_format: str,
            async_save: bool = False,
    ):
        super().__init__()
        self.save_ckpt_format = save_ckpt_format
        self.async_save = async_save

        self.save_sharded_strategy = self._determine_dist_ckpt_save_strategy()

    @debug_time('DistributedCheckpointIO.save_checkpoint')
    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        """ Saves a distributed checkpoint. Creates the checkpoint root directory if doesn't exist.

        Args:
            checkpoint (Dict[str, Any]): sharded state dict to save
            path (_PATH): checkpoint directory
            storage_options (Any, optional): Optional parameters when saving the checkpoint
        """
        fs = get_filesystem(path)
        fs.makedirs(path, exist_ok=True)

        dist_checkpointing.save(
            sharded_state_dict=checkpoint, checkpoint_dir=path, sharded_strategy=self.save_sharded_strategy
        )

    def maybe_finalize_save_checkpoint(self, blocking: bool = False) -> bool:
        if not self.async_save:
            return False
        with debug_time('DistributedCheckpointIO.maybe_finalize_save_checkpoint'):
            return self.save_sharded_strategy.maybe_finalize_async_save(blocking=blocking)

    @debug_time('DistributedCheckpointIO.load_checkpoint')
    def load_checkpoint(
        self, path: _PATH, map_location: Optional[Any] = None, sharded_state_dict: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """ Loads a distributed checkpoint.

        Args:
            path (_PATH): checkpoint directory
            map_location (Any, optional): required to be None in this implementation
            sharded_state_dict (Dict[str, Any], optional): state dict which
                defines the loading procedure for the distributed checkpoint.
                Defaults to None to comply with the CheckpointIO interface,
                but it's a required argument.

        Returns:
            Dist[str, Any]: loaded checkpoint.
        """
        if sharded_state_dict is None:
            raise ValueError('DistributedCheckpointIO requires passing sharded_state_dict argument to load_checkpoint')
        if map_location is not None:
            raise ValueError('DistributedCheckpointIO doesnt handle map_location argument')

        if self.save_ckpt_format == 'zarr':
            sharded_strategy = tensorstore.TensorStoreLoadShardedStrategy(load_directly_on_device=True)
        else:
            sharded_strategy = None

        return dist_checkpointing.load(
            sharded_state_dict=sharded_state_dict, checkpoint_dir=path, sharded_strategy=sharded_strategy
        )

    @debug_time('DistributedCheckpointIO.remove_checkpoint')
    def remove_checkpoint(self, path: _PATH) -> None:
        """ Remove a distributed checkpoint.

        Due to potentially large number of files, the implementation remove the whole directory at once.
        """
        shutil.rmtree(path, ignore_errors=True)

    def _determine_dist_ckpt_save_strategy(self):
        """ Determine the saving strategy based on storage config.

        For now only decides the checkpoint format.
        """
        save_strategy = (self.save_ckpt_format, 1)
        if self.async_save:
            if save_strategy[0] != 'torch_dist':
                raise ValueError('Async dist-ckpt save supported only for torch_dist format')
            save_strategy = TorchDistAsyncSaveShardedStrategy('torch_dist', 1)

        logging.info(f'Using {save_strategy} dist-ckpt save strategy.')
        return save_strategy
