from logging import getLogger
from typing import Any, Dict, Optional

from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest, debug_time
from torch.distributed.checkpoint._async_process_executor import _CheckpointRequestIdentifier
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.types import _PATH
from megatron.core.dist_checkpointing.mapping import apply_factories
from megatron.core.dist_checkpointing.state_dict_transformation import save_preprocess
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelSaveStrategyWrapper,
)
from nemo.utils import logging
from nemo.utils.callbacks.dist_ckpt_io import DistributedCheckpointIO

from nemo.utils._ckpt_utils import TorchCompatiblePersistentAsyncCaller
from nemo.utils._state_dict_utils import (
    _copy_state_dict,
    _create_cpu_state_dict,
    _offload_state_dict_to_cpu,
)
from nemo.utils.callbacks.daemon_strategies import DaemonTorchDistSaveShardedStrategy

logger = getLogger(__name__)


class MinCkptOverheadCheckpointIO(DistributedCheckpointIO):
    def __init__(
        self,
        save_ckpt_format: str,
        load_directly_on_device: bool = True,
        load_strictness: Optional["StrictHandling"] = None,
        async_save: bool = False,
        torch_dist_multiproc: Optional[int] = None,
        assume_constant_structure: bool = False,
        parallel_save: bool = False,
        parallel_save_within_dp: bool = False,
        parallel_load: bool = False,
    ):
        super().__init__(
            save_ckpt_format,
            load_directly_on_device,
            load_strictness,
            async_save,
            torch_dist_multiproc,
            assume_constant_structure,
            parallel_save,
            parallel_save_within_dp,
            parallel_load,
        )

        self._staged_state_dict = None

    @debug_time("OptimizedD2HCheckpointIO.save_checkpoint")
    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> Optional["AsyncRequest"]:
        """Saves a distributed checkpoint via checkpoint subprocess. Creates the checkpoint root directory if doesn't exist.

        Args:
            checkpoint (Dict[str, Any]): sharded state dict to save
            path (_PATH): checkpoint directory
            storage_options (Any, optional): Optional parameters when saving the checkpoint
        """
        fs = get_filesystem(path)
        fs.makedirs(path, exist_ok=True)

        validate_sharding_integrity = not (
            self.validated_consistency and self.assume_constant_structure
        )
        self.validated_consistency = True

        apply_factories(checkpoint)
        sharded_state_dict, common_state_dict = save_preprocess(
            checkpoint, validate_sharding_integrity, None
        )

        # TODO: Pin tensors to further improve d2h time
        common_state_dict = _offload_state_dict_to_cpu(common_state_dict)

        staged_state_dict = self._staged_state_dict
        if staged_state_dict is None:
            staged_state_dict = _create_cpu_state_dict(
                sharded_state_dict, pin_memory=True, share_memory=True
            )

        _copy_state_dict(sharded_state_dict, staged_state_dict, False)

        if self.assume_constant_structure:
            self._staged_state_dict = staged_state_dict

        staged_state_dict["common"] = common_state_dict
        staged_state_dict["thread_count"] = self.torch_dist_multiproc or 1

        def finalize_fn():
            staged_state_dict.pop("common")
            staged_state_dict.pop("thread_count")

        checkpoint_request_id = _CheckpointRequestIdentifier(path)

        self.async_request = AsyncRequest(
            async_fn=TorchCompatiblePersistentAsyncCaller._execute_save,
            async_fn_args=(
                staged_state_dict,
                checkpoint_request_id,
                self.save_sharded_strategy,
            ),
            finalize_fns=[finalize_fn],
        )

        return self.async_request

    def _determine_dist_ckpt_save_strategy(self):
        """Determine the saving strategy based on constructor args.

        Relies on the default MCore strategy unless extra PyT Distributed format arguments
        are passed in config or in case of a fully parallel save in which case
        a parallelization wrapper is applied.
        """
        if self.save_ckpt_format == "zarr":
            logging.warning(
                "`zarr` distributed checkpoint backend is deprecated."
                " Distributed optimizer checkpoint saving might be extremely slow."
                " Please switch to PyTorch Distributed format (model.dist_ckpt_format=torch_dist)."
            )

        if self.async_save and self.save_ckpt_format != "torch_dist":
            raise ValueError(
                "Async dist-ckpt save supported only for torch_dist format"
            )

        torch_dist_kwargs = (
            {}
            if self.torch_dist_multiproc is None
            else dict(thread_count=self.torch_dist_multiproc)
        )
        if self.save_ckpt_format == "torch_dist" and torch_dist_kwargs:
            save_strategy = DaemonTorchDistSaveShardedStrategy(
                self.save_ckpt_format, 1, **torch_dist_kwargs
            )
        else:
            save_strategy = get_default_save_sharded_strategy(self.save_ckpt_format, 1)

        # MCore v0.8 introduces `use_cached_ckpt_structure` attribute
        if hasattr(save_strategy, "use_cached_ckpt_structure"):
            save_strategy.use_cached_ckpt_structure = self.assume_constant_structure

        if self.parallel_save:
            parallelization_group = (
                get_data_parallel_group(with_context_parallel=True)
                if self.parallel_save_within_dp
                else None
            )
            save_strategy = FullyParallelSaveStrategyWrapper(
                save_strategy, parallelization_group, self.assume_constant_structure
            )

        logging.info(f"Using {save_strategy} dist-ckpt save strategy.")
        return save_strategy
