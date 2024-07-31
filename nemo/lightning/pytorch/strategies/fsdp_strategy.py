import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.strategies.fsdp import _get_sharded_state_dict_context
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
from pytorch_lightning.strategies.fsdp import FSDPStrategy as PLFSDPStrategy
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.distributed.checkpoint.state_dict import (  # get_state_dict,
    StateDictOptions,
    get_optimizer_state_dict,
    set_state_dict,
)
from torch.utils.data import DataLoader
from typing_extensions import override

from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.lightning import io
from nemo.lightning.io.pl import MegatronCheckpointIO
from nemo.lightning.megatron_parallel import masked_token_loss
from nemo.lightning.pytorch.strategies.utils import (
    ckpt_to_dir,
    mcore_to_pyt_sharded_state_dict,
    pyt_to_mcore_state_dict,
)
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO, AsyncFinalizerCallback


class FSDPStrategy(PLFSDPStrategy, io.IOMixin):
    """Megatron plugin for Pytorch Lightning.

    This strategy implements Fully-Sharded-Data-Parallel using PyTorch's native FSDP methods.
    Comparing with MegatronStrategy, FSDPStrategy is designed to be more lightweight, with
    minimal modifications over Lightning's FSDPStrategy but preserves necessary features to be
    compatible with nemo and mcore.

    Note:
        This strategy is designed to work with NVIDIA's Megatron-LM framework and requires
        specific model implementations that are compatible with Megatron's parallelism techniques.
    Note:
        Due to the different optimizer structure (FSDP only uses torch native optimizers),
        MegatronStrategy cannot resume training from checkpoints saved by FSDPStrategy, and vice
        versa. However, the model weights structure is made compatible, so switching strategy is
        possible if users only need the weights not the optimizer states. (E.g. run pretrain with
        megatron 4D parallelism and run SFT with FSDP.)
    """

    def __init__(self, *args, ckpt_include_optimizer=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ckpt_include_optimizer = ckpt_include_optimizer

    def _step_proxy(self, method_name, batch, batch_idx=None):
        if self.model != self.lightning_module:
            loss = self._forward_redirection(self.model, self.lightning_module, method_name, batch, batch_idx)
        else:
            loss = getattr(self.lightning_module, method_name)(batch, batch_idx)

        return masked_token_loss(loss, batch['loss_mask'])

    @override
    def training_step(self, batch, batch_idx=None) -> STEP_OUTPUT:
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.train_step_context():
            loss = self._step_proxy("training_step", batch, batch_idx)
            reduced_loss = average_losses_across_data_parallel_group([loss])

            self.lightning_module.log(
                'global_step',
                self.trainer.global_step,
                prog_bar=True,
                rank_zero_only=True,
                batch_size=1,
            )

            self.lightning_module.log(
                'step',
                self.trainer.global_step,
            )
            self.lightning_module.log(
                'reduced_train_loss', reduced_loss, prog_bar=True, rank_zero_only=True, batch_size=1
            )

            # returns unreduced loss for backward
            return loss

    @override
    def validation_step(self, batch, batch_idx=None) -> Any:
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.val_step_context():
            loss = self._step_proxy("validation_step", batch, batch_idx)
            reduced_loss = average_losses_across_data_parallel_group([loss])
            self.lightning_module.log('val_loss', reduced_loss, rank_zero_only=True, batch_size=1)
            return reduced_loss

    @override
    def test_step(self, batch, batch_idx=None) -> STEP_OUTPUT:
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.test_step_context():
            loss = self._step_proxy("test_step", batch, batch_idx)
            reduced_loss = average_losses_across_data_parallel_group([loss])
            return reduced_loss

    @override
    def predict_step(self, batch, batch_idx=None) -> STEP_OUTPUT:
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.predict_step_context():
            loss = self._step_proxy("predict_step", batch, batch_idx)
            reduced_loss = average_losses_across_data_parallel_group([loss])
            return reduced_loss

    @override
    def process_dataloader(self, dataloader: DataLoader) -> DataLoader:
        if self.data_sampler:
            return self.data_sampler.transform_dataloader(dataloader)

        return dataloader

    @property
    @override
    def checkpoint_io(self) -> CheckpointIO:
        # Taken from MegatronStrategy and made less configurable
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
        # Taken from MegatronStrategy
        if self.is_global_zero:
            shutil.rmtree(ckpt_to_dir(filepath))

    @override
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: Union[str, Path], storage_options: Optional[Any] = None
    ) -> None:
        """Converts PyT checkpoints to MCore format and save using MCore dist ckpt library."""
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
        """PTL method which we override to integrate distributed checkpoints for FSDP models.
        Different from MegatronStrategy, both model and optimizer states are restore within
        this method.

        The logic here is slightly more complicated:
        1. Obtain PyT state dicts (sharded & unflattened) for model and optim -> torch::ShardedTensor
        2. Convert to MCore state dicts -> mcore::ShardedTensor
        3. Load from checkpoint using MCore dist ckpt API -> torch::Tensor
        4. Convert to PyT state dicts (sharded & unflattened) -> torch::ShardedTensor
        5. Load into model and optim using PyT dist ckpt API
        6. Return the loaded checkpoint for lightning to load other metadata
        """
        path = Path(self.broadcast(checkpoint_path))
        torch.cuda.empty_cache()

        # TODO: the elegant way to load both state dicts. Need pytorch 2.3.1
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
            optim_state_dict=checkpoint['optimizer'] if self.ckpt_include_optimizer else None,
        )

        return checkpoint
