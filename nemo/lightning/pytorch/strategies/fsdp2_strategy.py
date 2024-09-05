# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import shutil
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import pytorch_lightning as pl
import torch
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning_fabric.plugins import CheckpointIO
from pytorch_lightning.strategies import ModelParallelStrategy, ParallelStrategy
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.distributed._composable.fsdp import FSDPModule, fully_shard
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.utils.data import DataLoader
from typing_extensions import override

from nemo.lightning import io
from nemo.lightning.pytorch.strategies.utils import (
    ckpt_to_dir,
    create_checkpoint_io,
    fix_progress_bar,
    init_model_parallel,
    mcore_to_pyt_sharded_state_dict,
    pyt_to_mcore_state_dict,
    setup_data_sampler,
    setup_parallel_ranks,
)


class FSDP2Strategy(ModelParallelStrategy, io.IOMixin):
    def __init__(
        self,
        replica_size: Union[Literal["auto"], int] = "auto",
        data_parallel_size: Union[Literal["auto"], int] = "auto",
        save_distributed_checkpoint: bool = True,
        ckpt_load_optimizer: bool = True,
        ckpt_save_optimizer: bool = True,
        data_sampler=None,
    ):
        ParallelStrategy.__init__(self)  # skip ModelParallelStrategy.__init__

        self._data_parallel_size = data_parallel_size
        self._replica_size = replica_size

        self._save_distributed_checkpoint = save_distributed_checkpoint
        self._process_group_backend: Optional[str] = None
        self._timeout: Optional[timedelta] = default_pg_timeout
        self._device_mesh: Optional[DeviceMesh] = None
        self.num_nodes = 1

        self.ckpt_load_optimizer = ckpt_load_optimizer
        self.ckpt_save_optimizer = ckpt_save_optimizer

        self.data_sampler = data_sampler

    @override
    def setup_environment(self) -> None:
        setup_parallel_ranks(self)
        ParallelStrategy.setup_environment(self)  # skip ModelParallelStrategy.setup_environment
        self._setup_distributed()
        if self._replica_size == "auto":
            self._replica_size = self.num_nodes
        if self._data_parallel_size == "auto":
            self._data_parallel_size = self.num_processes
        self._device_mesh = init_device_mesh(
            device_type=self.root_device.type,
            mesh_shape=(self._replica_size, self._data_parallel_size),
            mesh_dim_names=("replica", "data_parallel"),
        )
        init_model_parallel(self.model)

    @property
    @override
    def distributed_sampler_kwargs(self) -> Dict[str, Any]:
        assert self.device_mesh is not None
        return {"num_replicas": self.device_mesh.size(), "rank": self.device_mesh.get_rank()}

    @override
    def setup(self, trainer: pl.Trainer) -> None:
        self.trainer = trainer
        setup_data_sampler(self.trainer)
        fix_progress_bar(trainer)

        # injecting PEFT adapters
        if hasattr(self.model, "model_transform") and self.model.model_transform is not None:
            self.model.model_transform(self.model)

        # shard model
        from megatron.core.transformer.transformer_layer import TransformerLayer

        if not isinstance(self.model, FSDPModule):
            for sub_module in self.model.modules():
                if isinstance(sub_module, TransformerLayer):
                    fully_shard(sub_module, mesh=self.device_mesh)
            fully_shard(self.model, mesh=self.device_mesh)

        super().setup(trainer)

    def _get_loss_reduction(self, step_type: str):
        for fn_name in [f"{step_type}_loss_reduction", "loss_reduction"]:
            if hasattr(self.lightning_module, fn_name):
                return getattr(self.lightning_module, fn_name)
        return None

    def _step_proxy(self, step_type, batch, batch_idx=None):
        method_name = f"{step_type}_step"
        if self.model != self.lightning_module:
            loss = self._forward_redirection(self.model, self.lightning_module, method_name, batch, batch_idx)
        else:
            loss = getattr(self.lightning_module, method_name)(batch, batch_idx)

        _loss_reduction = self._get_loss_reduction(step_type)
        if _loss_reduction:
            return _loss_reduction.forward(batch, loss)
        return loss, {'avg': loss}

    @override
    def training_step(self, batch, batch_idx=None) -> STEP_OUTPUT:
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.train_step_context():
            loss, reduced = self._step_proxy("training", batch, batch_idx)

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
                'reduced_train_loss', reduced['avg'], prog_bar=True, rank_zero_only=True, batch_size=1
            )

            # returns unreduced loss for backward
            return loss

    @override
    def validation_step(self, batch, batch_idx=None) -> Any:
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.val_step_context():
            loss, reduced = self._step_proxy("validation", batch, batch_idx)
            self.lightning_module.log('val_loss', reduced['avg'], rank_zero_only=True, batch_size=1)
            return loss

    @override
    def test_step(self, batch, batch_idx=None) -> STEP_OUTPUT:
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.test_step_context():
            loss, reduced = self._step_proxy("test", batch, batch_idx)
            self.lightning_module.log('test_loss', reduced['avg'], rank_zero_only=True, batch_size=1)

            return loss

    @override
    def predict_step(self, batch, batch_idx=None) -> STEP_OUTPUT:
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.predict_step_context():
            loss, reduced = self._step_proxy("predict", batch, batch_idx)
            return reduced

    @override
    def process_dataloader(self, dataloader: DataLoader) -> DataLoader:
        if self.data_sampler:
            return self.data_sampler.transform_dataloader(dataloader)

        return dataloader

    @property
    @override
    def checkpoint_io(self) -> CheckpointIO:
        if not self._checkpoint_io:
            self._checkpoint_io = create_checkpoint_io()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: CheckpointIO) -> None:
        self._checkpoint_io = io

    @property
    def current_epoch_step(self) -> int:
        """
        Get the value of step within an epoch.
        """
        return max(
            self.trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.current.completed,
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.current.completed,
        )

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
        if "sharded_state_dict" not in checkpoint:
            checkpoint["sharded_state_dict"] = checkpoint.pop("state_dict")
        pyt_to_mcore_state_dict(checkpoint["sharded_state_dict"], device_mesh=self.device_mesh)
        checkpoint["state_dict"] = OrderedDict([])

        # TODO: do we still need to keep this?
        for optim_state in checkpoint['optimizer_states']:
            optim_state.pop("state")

        if self.trainer.state.fn == TrainerFn.FITTING and self.ckpt_save_optimizer:
            checkpoint['optimizer'] = get_optimizer_state_dict(
                self.model, self.optimizers, options=StateDictOptions(cpu_offload=True)
            )
            pyt_to_mcore_state_dict(
                checkpoint['optimizer']['state'], prefix="optimizer.state.", device_mesh=self.device_mesh
            )

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

        sharded_state_dict = {}

        msd, osd = get_state_dict(self.model, self.optimizers, options=StateDictOptions(cpu_offload=True))

        pyt_to_mcore_state_dict(msd, device_mesh=self.device_mesh)
        sharded_state_dict["sharded_state_dict"] = msd

        if self.ckpt_load_optimizer and self.trainer.state.fn == TrainerFn.FITTING:
            # osd = get_optimizer_state_dict(self.model, self.optimizers, options=StateDictOptions(cpu_offload=True))
            pyt_to_mcore_state_dict(osd['state'], prefix="optimizer.state.", device_mesh=self.device_mesh)
            sharded_state_dict["optimizer"] = osd

        checkpoint = self.checkpoint_io.load_checkpoint(path, sharded_state_dict=sharded_state_dict)
        mcore_to_pyt_sharded_state_dict(
            checkpoint['sharded_state_dict'], msd, dtensor=True, device_mesh=self.device_mesh
        )

        if self.ckpt_load_optimizer and self.trainer.state.fn == TrainerFn.FITTING:
            mcore_to_pyt_sharded_state_dict(
                checkpoint['optimizer']['state'], osd['state'], dtensor=True, device_mesh=self.device_mesh
            )

        set_state_dict(
            self.model,
            self.optimizers if self.ckpt_load_optimizer else [],
            model_state_dict=checkpoint['sharded_state_dict'],
            optim_state_dict=checkpoint['optimizer'] if self.ckpt_load_optimizer else None,
        )

        return checkpoint
