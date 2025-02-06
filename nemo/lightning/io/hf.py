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

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union

import lightning.pytorch as pl
import torch
from lightning.fabric.plugins import CheckpointIO
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.types import _PATH
from torch import nn
from typing_extensions import Self, override

from nemo.lightning.ckpt_utils import WEIGHTS_PATH, ckpt_to_dir
from nemo.lightning.io.mixin import IOMixin

log = logging.getLogger(__name__)


LightningModuleT = TypeVar("LightningModuleT", bound=pl.LightningModule)
ModuleT = TypeVar("ModuleT", bound=nn.Module)


class HFCheckpointIO(CheckpointIO, IOMixin):
    """HFCheckpointIO that utilizes :func:`torch.save` and :func:`torch.load` to save and load checkpoints respectively,
    common for most use cases.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    """

    def __init__(self, model=None, save_adapter_only=False):
        super().__init__()
        self.save_adapter_only = save_adapter_only
        self.model = model

    @override
    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            path: write-target path
            storage_options: not used in ``TorchCheckpointIO.save_checkpoint``

        Raises
        ------
            TypeError:
                If ``storage_options`` arg is passed in

        """
        if self.save_adapter_only:
            return self._save_adapter_weights_only(checkpoint, path, storage_options)
        elif callable(getattr(self.model, 'save_pretrained', None)):
            return self.model.save_pretrained(path, state_dict=checkpoint.pop('state_dict'))
        else:
            return super().save_checkpoint(checkpoint, path, storage_options)

    def _save_adapter_weights_only(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        from safetensors.torch import save_file

        state_dict = {}
        module_names = list(checkpoint["state_dict"].keys())
        for name in module_names:
            param = checkpoint["state_dict"].pop(name)
            name = name\
                .replace("model.model", "base_model.model")\
                .replace("lora_a.weight", "lora_A.weight")\
                .replace("lora_b.weight", "lora_B.weight")
            state_dict[name] = param

        checkpoint_dir = ckpt_to_weights_subdir(path, is_saving=True)
        fs = get_filesystem(checkpoint_dir)
        fs.makedirs(checkpoint_dir, exist_ok=True)
        save_file(state_dict, checkpoint_dir / "adapter_model.safetensors")

    @override
    def load_checkpoint(
        self,
        path: _PATH,
        sharded_state_dict=None,
        map_location: Optional[Callable] = None,
        strict: Optional['StrictHandling'] | bool = None,  # noqa: F821
    ) -> Dict[str, Any]:
        """Loads checkpoint using :func:`torch.load`, with additional handling for ``fsspec`` remote loading of files.

        Args:
            path: Path to checkpoint
            map_location: a function, :class:`torch.device`, string or a dict specifying how to remap storage
                locations.

        Returns: The loaded checkpoint.

        Raises
        ------
            FileNotFoundError: If ``path`` is not found by the ``fsspec`` filesystem

        """

        # Try to read the checkpoint at `path`. If not exist, do not restore checkpoint.
        fs = get_filesystem(path)
        if not fs.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        if not fs.isdir(path):
            raise ValueError(
                f"Checkpoints should be a directory. Found: {path}.")

        state_dict = None
        if (path / "adaptor_config.json").exists():
            from safetensors import safe_open

            state_dict = {}
            with safe_open("adapter_model.safetensors", framework="pt", device=0) as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)

        return {'state_dict': state_dict}

    @override
    def remove_checkpoint(self, path: _PATH) -> None:
        """Remove checkpoint file from the filesystem.

        Args:
            path: Path to checkpoint

        """
        fs = get_filesystem(path)
        if fs.exists(path):
            fs.rm(path, recursive=True)
            log.debug(f"Removed checkpoint: {path}")
