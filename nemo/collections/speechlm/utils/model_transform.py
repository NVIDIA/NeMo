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

import json
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import lightning.pytorch as pl
import torch
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities import model_summary
from typing_extensions import override

from nemo.collections.llm.fn import base as fn
from nemo.lightning.ckpt_utils import ADAPTER_META_FILENAME
from nemo.lightning.io.pl import ckpt_to_dir, ckpt_to_weights_subdir
from nemo.lightning.megatron_parallel import MegatronParallel
from nemo.lightning.pytorch.callbacks.peft import PEFT
from nemo.utils import logging
from nemo.utils.callbacks.dist_ckpt_io import AsyncCompatibleCheckpointIO


class SpeechToTextLLMPEFT(PEFT):
    def __init__(self, peft: PEFT):
        super().__init__()
        self.peft = peft

    @override
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """PTL callback setup function."""
        from nemo.lightning.pytorch.strategies.utils import create_checkpoint_io

        super(PEFT, self).setup(trainer, pl_module, stage=stage)

        trainer.strategy.trainer = trainer
        wrapped_io = partial(WrappedAdapterIO, peft=self)

        ckpt_io_kwarg_names = [
            "save_ckpt_format",
            "async_save",
            "torch_dist_multiproc",
            "assume_constant_structure",
            "parallel_save",
            "parallel_save_within_dp",
            "parallel_load",
            "load_directly_on_device",
        ]
        ckpt_io_kwargs = {
            arg: getattr(trainer.strategy, arg)
            for arg in filter(lambda x: hasattr(trainer.strategy, x), ckpt_io_kwarg_names)
        }
        trainer.strategy._checkpoint_io = create_checkpoint_io(wrapping_ckpt_io=wrapped_io, **ckpt_io_kwargs)
        self.wrapped_io = (
            trainer.strategy._checkpoint_io._checkpoint_io
            if getattr(trainer.strategy, 'async_save', False)
            else trainer.strategy._checkpoint_io
        )
        trainer.strategy._init_model_parallel = False
        trainer.strategy._setup_optimizers = False

    def __call__(
        self, model: "nemo.collections.slm.model.SpeechToTextLLM"
    ) -> "nemo.collections.slm.model.SpeechToTextLLM":
        """Apply the PEFT method to the entire model.

        This method freezes the LLM parameters and walks through the model
        structure, applying the transform method to each module.

        Args:
            model (nn.Module): The model to be fine-tuned.

        Returns:
            nn.Module: The transformed model with PEFT applied.
        """
        # only apply PEFT to the language model
        model.freeze_llm()
        module = model.module
        logging.info(f"Applying PEFT to language model with: {self.peft}")

        # If using megatron virtual pipeline parallelism, model is a list of
        # model chunks so iterate over model
        if isinstance(model, MegatronParallel) and len(model) > 1:
            for model_chunk in model:
                self._transform_module(model_chunk)
        else:
            self._transform_module(module)

        if hasattr(model, "trainer") and model.trainer.state.fn != TrainerFn.FITTING:
            self.freeze_model(model)

        logging.info(f"\n{model_summary.summarize(model, max_depth=3)}")
        return model

    def _transform_module(self, module):
        while not hasattr(module, "language_model"):
            module = module.module
        fn.walk(module.language_model, self.transform, _skip_map=True)

    def transform(self, module, name=None, prefix=None):
        return self.peft.transform(module, name=name, prefix=prefix)


class WrappedAdapterIO(_WrappingCheckpointIO, AsyncCompatibleCheckpointIO):
    peft: Optional[PEFT] = None
    model_ckpt_path: Optional[Path] = None
    adapter_ckpt_path: Optional[Path] = None

    def __init__(self, checkpoint_io: Optional["CheckpointIO"] = None, peft: Optional[PEFT] = None) -> None:
        self.peft = peft
        super().__init__(checkpoint_io)

    @override
    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        assert self.checkpoint_io is not None

        state_key = 'sharded_state_dict'
        if not state_key in checkpoint:
            state_key = 'state_dict'
        checkpoint[state_key] = dict(
            filter(lambda item: self.peft.adapter_key_filter(item[0]), checkpoint[state_key].items())
        )
        request = self.checkpoint_io.save_checkpoint(checkpoint, path, storage_options=storage_options)

        from nemo.utils.get_rank import is_global_rank_zero

        if is_global_rank_zero():
            metadata = {"model_ckpt_path": str(self.model_ckpt_path)}
            base_dir = ckpt_to_weights_subdir(path, is_saving=True)
            base_dir.mkdir(parents=True, exist_ok=True)
            adapter_meta_path = base_dir / ADAPTER_META_FILENAME
            with open(adapter_meta_path, "w") as f:
                json.dump(metadata, f)
        return request

    @override
    def load_checkpoint(
        self, path: _PATH, sharded_state_dict=None, map_location: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        =====================
        Initial PEFT Training
        =====================
        Initial PEFT training requires loading the base model weights. In this case, this function is called by
        trainer.strategy.setup() -> megatron_strategy.restore_model() -> megatron_strategy.load_checkpoint().
        `path = PosixPath(<base_path>)`, and sharded_state_dict contains only base model weights

        ===========
        PEFT Resume
        ===========
        PEFT resume requires loading two set of model weights, 1) base model weights and 2) adapter weights
        Base model weights could be imported from e.g. HF, and is frozen during PEFT training.
        Adapter weights contains the training metadata that will need to be loaded.
        As such, this function will be entered twice during PEFT training resume.

        For the FIRST TIME this function is called by trainer._checkpoint_connector._restore_modules_and_callbacks.
        `path = AdapterPath(<adapter_path>, base_model_path=<base_path>)`, and sharded_state_dict contains only base model weights

        For the SECOND TIME this function is called by PEFT.apply_transform (above, in the same file).
        `path = PosixPath(<adapter_path>)`, and sharded_state_dict contains only adapter weights.
        """

        assert self.checkpoint_io is not None

        adapter_meta_path = ckpt_to_dir(path) / ADAPTER_META_FILENAME
        adapter_ckpt = None
        load_base = False

        if getattr(path, "base_model_path", None):
            ## PEFT Resume, FIRST TIME
            self.adapter_ckpt_path = Path(str(path))
            adapter_ckpt = self.checkpoint_io.load_checkpoint(path)  # Loads only metadata
            # path is adapter path to restore the training metadata, but switch to loading base model here.
            path = self.model_ckpt_path = path.base_model_path
            load_base = True
        elif adapter_meta_path.exists():
            ## PEFT Resume, SECOND TIME
            with open(adapter_meta_path, "r") as f:
                metadata = json.load(f)
            self.model_ckpt_path = Path(metadata['model_ckpt_path'])
            self.adapter_ckpt_path = path
        else:
            ## Initial PEFT Training
            self.model_ckpt_path = path

        # Note: this will include the Trainer-state of the model-checkpoint
        model_ckpt = self._load_checkpoint(path, sharded_state_dict, map_location, load_base)

        if adapter_ckpt is not None:
            ## PEFT Resume, FIRST TIME
            adapter_ckpt['state_dict'].update(model_ckpt['state_dict'])
            return adapter_ckpt
        return model_ckpt

    def _load_checkpoint(
        self, path: _PATH, sharded_state_dict, map_location: Optional[Callable] = None, load_base: bool = False
    ) -> None:
        if load_base:
            import tempfile

            with tempfile.TemporaryDirectory() as tempdir:
                # base_sharded_state_dict = self._get_base_sharded_state_dict(sharded_state_dict)
                base_sharded_state_dict = {
                    "state_dict": dict(),
                    "sharded_state_dict": dict(sharded_state_dict['state_dict']),
                }
                # retrieve `sharded_state_dict` if it has not already been configured in `on_save_checkpoint`
                async_save = getattr(self.checkpoint_io, "async_save", False)
                self.checkpoint_io.async_save = False
                self.checkpoint_io.save_checkpoint(base_sharded_state_dict, tempdir)
                model_ckpt = self.checkpoint_io.load_checkpoint(tempdir, sharded_state_dict, map_location)
                model_ckpt = self._fix_ckpt_device(model_ckpt)
                self.checkpoint_io.async_save = async_save
                torch.cuda.empty_cache()
            return model_ckpt
        else:
            model_ckpt = self.checkpoint_io.load_checkpoint(path, sharded_state_dict, map_location)

        return model_ckpt

    def _fix_ckpt_device(self, ckpt: Dict[str, Any]) -> Dict[str, Any]:
        assert torch.cuda.is_initialized(), (torch.cuda.is_available(), torch.cuda.is_initialized())
        cur_dev = torch.device("cuda", index=torch.cuda.current_device())
        from megatron.core.dist_checkpointing.dict_utils import dict_list_map_outplace

        def _fix_device(t):
            if isinstance(t, torch.Tensor) and t.device != cur_dev:
                t = t.to(cur_dev)
            return t

        return dict_list_map_outplace(_fix_device, ckpt)
