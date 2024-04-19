import shutil
from typing import Optional, Any, Dict

from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.types import _PATH
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.strategies import tensorstore

from nemo.utils import logging
from nemo.utils.model_utils import ckpt_to_dir


class DistributedCheckpointIO(CheckpointIO):
    def __init__(self, save_ckpt_format):
        super().__init__()
        self.save_ckpt_format = save_ckpt_format

        self.save_sharded_strategy = self.determine_dist_ckpt_save_strategy()

    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH,
                        storage_options: Optional[Any] = None) -> None:

        # dist_checkpointing expects a directory so we will name the directory
        # using the path with the file extension removed
        checkpoint_dir = ckpt_to_dir(path)
        fs = get_filesystem(checkpoint_dir)
        fs.makedirs(checkpoint_dir, exist_ok=True)

        dist_checkpointing.save(
            sharded_state_dict=checkpoint,
            checkpoint_dir=checkpoint_dir,
            sharded_strategy=self.save_sharded_strategy
        )

    def load_checkpoint(self, path: _PATH,
                        map_location: Optional[Any] = None,
                        sharded_state_dict: dict = None) -> Dict[str, Any]:
        if sharded_state_dict is None:
            raise ValueError('DistributedCheckpointIO requires passing sharded_state_dict argument to load_checkpoint')
        if map_location is not None:
            raise ValueError('DistributedCheckpointIO doesnt handle map_location argument')

        if self.save_ckpt_format == 'zarr':
            sharded_strategy = tensorstore.TensorStoreLoadShardedStrategy(load_directly_on_device=True)
        else:
            sharded_strategy = None

        return dist_checkpointing.load(
            sharded_state_dict=sharded_state_dict, checkpoint_dir=path,
            sharded_strategy=sharded_strategy
        )

    def remove_checkpoint(self, path: _PATH) -> None:
        shutil.rmtree(ckpt_to_dir(path), ignore_errors=True)

    def determine_dist_ckpt_save_strategy(self):
        """ Determine the saving strategy based on storage config.

        For now only decides the checkpoint format.
        """
        save_strategy = (self.save_ckpt_format, 1)
        logging.info(f'Using {save_strategy} dist-ckpt save strategy.')
        return save_strategy

    def teardown(self) -> None:
        super().teardown()