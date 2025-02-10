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
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

import lightning.pytorch as pl
import torch
from lightning.fabric.plugins import CheckpointIO
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.types import _PATH
from torch import nn
from typing_extensions import override
from nemo.lightning.io.pl import ckpt_to_weights_subdir
from nemo.lightning.io.mixin import IOMixin
import torch.distributed as dist
from nemo.lightning.ckpt_utils import WEIGHTS_PATH

log = logging.getLogger(__name__)


LightningModuleT = TypeVar("LightningModuleT", bound=pl.LightningModule)
ModuleT = TypeVar("ModuleT", bound=nn.Module)

def is_rank_0():
    """ Checks whether rank=0 accounting for un-inintialized dist-env"""
    return not dist.is_available() \
        or not dist.is_initialized() \
        or dist.get_rank() == 0

class HFCheckpointIO(CheckpointIO, IOMixin):
    """HFCheckpointIO that utilizes :func:`torch.save` and :func:`torch.load` to save and load
    checkpoints respectively, common for most use cases.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    """

    def __init__(self, model=None, adapter_only=False):
        """Initializes HFCheckpointIO

        Args:
            model (nn.Module, optional): The nn.Module that's used for training.
                This supplies the save_pretrained function.
            adapter_only (bool, optional): If true, will only save LoRA adapter weights. Defaults to False.
        """
        super().__init__()
        self.adapter_only = adapter_only
        self.model = model

    @override
    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        """
        Save model/training states to a checkpoint file.

        Note:
            This function assumes it's only written by RANK=0 if executed inside a dist-env.
        Args:
            checkpoint: dict containing model and trainer state
            path: write-target path
            storage_options: not used in ``TorchCheckpointIO.save_checkpoint``

        Raises
        ------
            TypeError:
                If ``storage_options`` arg is passed in

        """
        assert is_rank_0(), "Expected to run only on rank=0"
        # Determine checkpoint directory & make dir
        checkpoint_dir = ckpt_to_weights_subdir(path, is_saving=True)
        fs = get_filesystem(checkpoint_dir)
        fs.makedirs(checkpoint_dir, exist_ok=True)

        assert checkpoint_dir.parts[-1] == WEIGHTS_PATH, \
            "Expected % to end with %".format(checkpoint_dir, WEIGHTS_PATH)

        if self.adapter_only:
            # In this case the output looks like the following:
            # default--reduced_train_loss=0.0112-epoch=2-step=3
            # ├── context
            # │   ├── 0b9ee504-0ab7-4470-911b-cf7fc0223cde
            # │   ├── io.json
            # │   └── model.yaml
            # ├── trainer.pt
            # └── weights
            #     ├── adapter_config.json
            #     └── adapter_model.safetensors
            # Where the `trainer.pt` stores trainer's state (optimizer, dataloader, etc).
            # The `weights` directory contains the adapter's state dict, in HF format.
            self._save_adapter_weights_only(checkpoint.pop(
                'state_dict'), checkpoint_dir, storage_options)
            torch.save(checkpoint, checkpoint_dir.parent / 'trainer.pt')
        elif callable(getattr(self.model, 'save_pretrained', None)):
            # In this case the output looks like the following:
            # default--reduced_train_loss=0.0112-epoch=2-step=3
            # ├── weights
            # │   ├── config.json
            # │   ├── generation_config.json
            # │   ├── model.safetensors
            # │   ├── special_tokens_map.json
            # │   ├── tokenizer.json
            # │   └── tokenizer_config.json
            # └── trainer.pt
            # Where the `weights` directory contains the model's state dict, in HF format.
            # The `trainer.pt` stores trainer's state (optimizer, dataloader, etc).
            self.model.save_pretrained(
                checkpoint_dir, state_dict=checkpoint.pop('state_dict'))
            torch.save(checkpoint, checkpoint_dir.parent / 'trainer.pt')
        else:
            super().save_checkpoint(checkpoint, path, storage_options)
            raise NotImplementedError("Checkpoint was saved at: " + str(path))

    def _save_adapter_weights_only(
        self, state_dict: Dict[str, Any], path: Union[str, Path], storage_options: Optional[Any] = None
    ) -> None:
        """
        Saves only the adapter weights in a safetensors format.

        Args:
            state_dict (Dict[str, Any]): The state dictionary containing model weights.
            path (Union[str, Path]): The directory path where the adapter weights should be saved.
            storage_options (Optional[Any], optional): Additional storage options, if required.

        Raises:
            OSError: If saving the file fails.
        """
        from safetensors.torch import save_file

        # Rename keys in state_dict to match expected format
        module_names = list(state_dict.keys())
        for name in module_names:
            param = state_dict.pop(name)
            name = name\
                .replace("model.model", "base_model.model")\
                .replace("lora_a.weight", "lora_A.weight")\
                .replace("lora_b.weight", "lora_B.weight")
            state_dict[name] = param

        # Save weights to safetensors format
        try:
            save_file(state_dict, path / "adapter_model.safetensors")
        except OSError as e:
            raise OSError("Failed to save adapter weights: %".format(e))

    @staticmethod
    def _load_adapter_weights_only(path: Union[str, Path]) -> Dict[str, Any]:
        """
        Loads only the adapter weights from a safetensors checkpoint.

        Args:
            path (Union[str, Path]): The directory path where the adapter weights are stored.

        Returns:
            Dict[str, Any]: A dictionary containing the state dictionary of the adapter model.

        Raises:
            FileNotFoundError: If the checkpoint directory does not exist.
            ValueError: If the checkpoint path is not a directory.
            OSError: If loading the weights fails.
        """
        fs = get_filesystem(path)

        if not fs.exists(path):
            raise FileNotFoundError("Checkpoint file not found: %", path)

        if not fs.isdir(path):
            raise ValueError("Checkpoints should be a directory. Found: %", path)

        state_dict = {}
        adapter_file = Path(path) / "adapter_model.safetensors"
        if not adapter_file.exists():
            raise FileNotFoundError("Adapter weights file not found: %", adapter_file)
        config_file = Path(path) / "adapter_config.json"
        if not config_file.exists():
            raise FileNotFoundError("Adapter config file not found: %", config_file)

        from safetensors import safe_open
        try:
            with safe_open(adapter_file, framework="pt", device=0) as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        except OSError as e:
            raise OSError(f"Failed to load adapter weights: {e}")

        return {'state_dict': state_dict}

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
        path = Path(path)
        assert path.parts[-1] == WEIGHTS_PATH, "Expected % to end with %".format(path, WEIGHTS_PATH)
        trainer_state = {}

        if not (path.parent / 'trainer.pt').exists():
            logging.info("Asked to restore from checkpoint without trainer state at %", path)
        else:
            trainer_state = torch.load(
                path.parent / 'trainer.pt',
                map_location='cpu',
                mmap=True,
                weights_only=False,
            )

        if self.adapter_only:
            trainer_state |= HFCheckpointIO._load_adapter_weights_only(path)
        elif callable(getattr(self.model, 'load_pretrained', None)):
            trainer_state['state_dict'] = self.model.load_pretrained(
                f'{path}/model/')
        else:
            raise ValueError("Badio")

        return trainer_state

    @override
    def remove_checkpoint(self, path: _PATH) -> None:
        """Remove checkpoint file from the filesystem.

        Args:
            path: Path to checkpoint

        """
        fs = get_filesystem(path)
        if fs.exists(path):
            fs.rm(path, recursive=True)
            log.debug("Removed checkpoint: %s", path)
