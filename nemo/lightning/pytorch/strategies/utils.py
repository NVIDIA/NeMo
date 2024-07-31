import io
from typing import Any, Dict, Iterable, List, Tuple, Union
from pathlib import Path

import torch
from torch.distributed._sharded_tensor import ShardedTensor as TorchShardedTensor

from megatron.core import parallel_state
from megatron.core.transformer.utils import _get_extra_state_offsets
from megatron.core.dist_checkpointing.mapping import (
    ShardedBase,
    ShardedObject,
    ShardedTensor,
)
from megatron.core.dist_checkpointing.strategies.torch import sharded_tensor_to_torch_sharded_tensor


def ckpt_to_dir(filepath: Union[str, Path]) -> Path:
    """PTL considers checkpoints as .ckpt files.
    This method removes the extension and returns a path
    to be used as a directory for distributed checkpoints.
    """
    filepath = Path(filepath)

    if filepath.suffix == ".ckpt":
        return filepath.with_name(filepath.stem)

    return filepath


def mcore_to_pyt_sharded_state_dict(
    checkpoint: Dict[str, List[torch.Tensor]],
    sharded_state_dict: Dict[str, Union[List[ShardedTensor], ShardedObject]],
) -> Dict[str, Union[TorchShardedTensor, io.BytesIO]]:

    def _mcore_to_pyt_sharded_tensor(
        tens: List[torch.Tensor],
        sh_tens: List[ShardedTensor],
    ) -> TorchShardedTensor:
        for ten, sh_ten in zip(tens, sh_tens):
            # remove prepend axes and put in loaded tensor
            sh_ten.global_shape = sh_ten.global_shape[sh_ten.prepend_axis_num:]
            sh_ten.global_offset = sh_ten.global_offset[sh_ten.prepend_axis_num:]
            sh_ten.axis_fragmentations = sh_ten.axis_fragmentations[sh_ten.prepend_axis_num:]
            sh_ten.prepend_axis_num = 0
            sh_ten.data = ten
            sh_ten.validate_metadata_integrity()

        return sharded_tensor_to_torch_sharded_tensor(sh_tens)

    def _convert(checkpoint, sharded_state_dict, k):
        assert k in sharded_state_dict, f"{k} not in sharded_state_dict"

        if isinstance(sharded_state_dict[k], Dict):
            for kk in sharded_state_dict[k]:
                _convert(checkpoint[k], sharded_state_dict[k], kk)
        elif isinstance(sharded_state_dict[k], ShardedObject):
            """Do nothing. checkpoint[k] contains loaded io.BytesIO already."""
        elif isinstance(sharded_state_dict[k], List):   # list of ShardedTensor
            checkpoint[k] = _mcore_to_pyt_sharded_tensor(checkpoint[k], sharded_state_dict[k])
    
    for k in checkpoint:
        _convert(checkpoint, sharded_state_dict, k)

    return checkpoint
                   

def pyt_to_mcore_state_dict(
    state_dict: Dict[str, Any],
    prefix: str = ""
) -> Dict[str, List[ShardedBase]]:

    def _torch_to_mcore_sharded_tensor(
        key: str, 
        sh_ten: TorchShardedTensor, 
        prepend_offsets: Iterable[Tuple[int, int, int]] = (),
        prefix: str = "",
        allow_shape_mismatch: bool = False
    ) -> List[ShardedTensor]:
        prepend_axis_num = len(prepend_offsets)
        
        assert isinstance(sh_ten, TorchShardedTensor), sh_ten
        sharded_meta = sh_ten.metadata()
        local_shards = sh_ten.local_shards()

        # DEBUG
        assert all([ls.metadata.placement == local_shards[0].metadata.placement for ls in local_shards]), [ls.meta.placement for ls in local_shards]

        global_shape = sharded_meta.size

        axis = list(range(len(global_shape)))
        axis_fragm = [global_shape[i] // local_shards[0].metadata.shard_sizes[i] for i in axis]
        rank_offsets = []

        for i in range(len(local_shards)):
            local_shard = local_shards[i]
            ten, meta = local_shard.tensor, local_shard.metadata

            for j in range(len(axis)):
                axis_rank_offset = meta.shard_offsets[j] // meta.shard_sizes[j]
                rank_offsets.append((axis[j] + prepend_axis_num, axis_rank_offset, axis_fragm[j]))

            local_shards[i] = ShardedTensor.from_rank_offsets(
                f"{prefix}{key}",
                ten,
                *prepend_offsets, # prepend layer shards
                *rank_offsets,
                replica_id=0,
                prepend_axis_num=prepend_axis_num,
                allow_shape_mismatch=allow_shape_mismatch,
            )

        return local_shards
    
    def _torch_to_mcore_sharded_object(
        key: str,
        obj: io.BytesIO,
        sharded_offsets: Iterable[Tuple[int, int, int]] = (),
        prefix: str = "",
    ) -> ShardedObject:
        replica_id = (
            0,
            0,
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )

        return ShardedObject(
            f"{prefix}{key}", 
            obj, 
            *_get_extra_state_offsets(sharded_offsets), 
            replica_id
        )

    def _convert(state_dict, k, sh_key, v, prepend_offsets, prefix="", allow_shape_mismatch=False):
        if isinstance(v, Dict):
            for kk, vv in v.items():
                _convert(
                    v, 
                    kk, 
                    sh_key, 
                    vv, 
                    prepend_offsets, 
                    prefix=f"{prefix}{kk}.",
                    allow_shape_mismatch=allow_shape_mismatch
                )
        elif isinstance(v, TorchShardedTensor):
            state_dict[k] = _torch_to_mcore_sharded_tensor(
                sh_key, 
                v, 
                prepend_offsets, 
                prefix=prefix, 
                allow_shape_mismatch=allow_shape_mismatch
            )
        elif isinstance(v, io.BytesIO):
            state_dict[k] = _torch_to_mcore_sharded_object(sh_key, v, prepend_offsets, prefix)


    num_layers = 0
    for k in state_dict:
        if k.startswith("module.decoder.layers."):
                num_layers = max(num_layers, int(k.split('.')[3]) + 1)
    assert num_layers != 0

    for k, v in state_dict.items():
        prepend_offsets = []
        sh_key = k
        allow_shape_mismatch = k.endswith(".word_embeddings.weight")    # vocab size can be different
        if k.startswith("module.decoder.layers."):
            sh_key = k.split('.')
            global_layer_offset = int(sh_key.pop(3))
            sh_key = '.'.join(sh_key)
            prepend_offsets.append((0, global_layer_offset, num_layers))

        _convert(state_dict, k, sh_key, v, prepend_offsets, prefix, allow_shape_mismatch)

    return state_dict
