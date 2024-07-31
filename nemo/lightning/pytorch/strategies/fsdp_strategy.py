from collections import OrderedDict
from typing import Any, Dict, Union, Optional
from typing_extensions import override
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    # get_state_dict, 
    set_state_dict,
    StateDictOptions,
)

from pytorch_lightning.strategies.fsdp import FSDPStrategy as PLFSDPStrategy
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.strategies.fsdp import _get_sharded_state_dict_context

from nemo.lightning.io.pl import MegatronCheckpointIO
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO, AsyncFinalizerCallback

from nemo.lightning.pytorch.strategies.utils import (
    ckpt_to_dir,
    pyt_to_mcore_state_dict,
    mcore_to_pyt_sharded_state_dict,
)


class FSDPStrategy(PLFSDPStrategy):
    def __init__(self, *args,  ckpt_include_optimizer=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ckpt_include_optimizer = ckpt_include_optimizer

    @override
    def process_dataloader(self, dataloader: DataLoader) -> DataLoader:
        if self.data_sampler:
            return self.data_sampler.transform_dataloader(dataloader)
        
        return dataloader
    
    @property
    @override
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            checkpoint_callback = self.trainer.checkpoint_callback
            async_save = getattr(checkpoint_callback, "async_save", False)
            self._checkpoint_io = MegatronCheckpointIO(async_save=async_save)
            if async_save:
                self._checkpoint_io = AsyncFinalizableCheckpointIO(self._checkpoint_io)
                have_async_callback = False
                for callback in self.trainer.callbacks:
                    if isinstance(callback, AsyncFinalizerCallback):
                        have_async_callback = True
                        break
                if not have_async_callback:
                    self.trainer.callbacks.append(AsyncFinalizerCallback())
        elif isinstance(self._checkpoint_io, _WrappingCheckpointIO):
            self._checkpoint_io.checkpoint_io = MegatronCheckpointIO()

        return self._checkpoint_io
    
    @override
    def remove_checkpoint(self, filepath: Union[str, Path]) -> None:
        if self.is_global_zero:
            shutil.rmtree(ckpt_to_dir(filepath))
    
    @override
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: Union[str, Path], storage_options: Optional[Any] = None
    ) -> None:
        checkpoint["sharded_state_dict"] = pyt_to_mcore_state_dict(checkpoint.pop("state_dict"))
        checkpoint["state_dict"] = OrderedDict([])

        # TODO: do we still need to keep this?
        for optim_state in checkpoint['optimizer_states']:
            optim_state.pop("state")

        if self.trainer.state.fn == TrainerFn.FITTING and self.ckpt_include_optimizer:
            checkpoint['optimizer'] = get_optimizer_state_dict(self.model, self.optimizers)
            pyt_to_mcore_state_dict(checkpoint['optimizer']['state'], prefix="optimizer.state.")

        self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)

    @override
    def load_checkpoint(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        path = Path(self.broadcast(checkpoint_path))
        torch.cuda.empty_cache()

        # TODO: the elegant way to load both state_dict. Need pytorch 2.3.1
        # msd, osd = get_state_dict(self.model, self.optimizers, options=StateDictOptions(cpu_offload=True))
        sharded_state_dict = {}
        with _get_sharded_state_dict_context(self.model):
            msd = self.model.state_dict()
            pyt_to_mcore_state_dict(msd)
            sharded_state_dict["sharded_state_dict"] = msd

        if self.ckpt_include_optimizer and self.trainer.state.fn == TrainerFn.FITTING:
            osd = get_optimizer_state_dict(self.model, self.optimizers, options=StateDictOptions(cpu_offload=True))
            pyt_to_mcore_state_dict(osd['state'], prefix="optimizer.state.")
            sharded_state_dict["optimizer"] = osd


        checkpoint = self.checkpoint_io.load_checkpoint(path, sharded_state_dict=sharded_state_dict)
        mcore_to_pyt_sharded_state_dict(checkpoint['sharded_state_dict'], msd)
    
        if self.ckpt_include_optimizer and self.trainer.state.fn == TrainerFn.FITTING:
            mcore_to_pyt_sharded_state_dict(checkpoint['optimizer']['state'], osd['state'])

        set_state_dict(
            self.model, 
            self.optimizers if self.ckpt_include_optimizer else [], 
            model_state_dict=checkpoint['sharded_state_dict'],
            optim_state_dict=checkpoint['optimizer'] if self.ckpt_include_optimizer else None
        )
        
        return checkpoint
