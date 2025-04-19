# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import re
from contextlib import contextmanager
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import yaml
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.validation import StrictHandling, parse_strict_flag
from megatron.core.transformer import TransformerConfig, TransformerLayer

from nemo import lightning as nl
from nemo.collections import llm
from nemo.utils import logging
from nemo.utils.import_utils import safe_import, safe_import_from

from .loss import HiddenStateCosineLoss, LogitsAndIntermediatesLossBalancer, LogitsKLLoss, ProjectionLayer

if TYPE_CHECKING:
    from megatron.core.dist_checkpointing.mapping import ShardedStateDict
    from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

mto, _ = safe_import("modelopt.torch.opt")
DistillationModel, _ = safe_import_from("modelopt.torch.distill", "DistillationModel", alt=object)
DistillationLossBalancer, _ = safe_import_from("modelopt.torch.distill", "DistillationLossBalancer", alt=object)


def load_distillation_config(
    config_path: Optional[str], student_cfg: TransformerConfig, teacher_cfg: TransformerConfig
) -> Dict[str, Any]:
    """Read the distillation yaml config file specified by ``args.export_kd_cfg``.

    Args:
        config_path: Path to user-defined distillation settings yaml file.
            If `None`, uses default logits-only distillation mode for GPT models.
        student_cfg: Model config for student model.
        teacher_cfg: Model config for teacher model.

    WARNING: Assumes intermediate hidden sizes are always that found in the model config's ``hidden_size`` attribute.
    """
    if not config_path:
        logging.warning("Distillation config not provided. Using default.")
        cfg = {
            "logit_layers": ["output_layer", "output_layer"],
            "intermediate_layer_pairs": [],
            "skip_lm_loss": True,
            "kd_loss_scale": 1.0,
        }
    else:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

    intermediate_pairs = cfg["intermediate_layer_pairs"]
    logit_pair = cfg["logit_layers"]
    skip_lm_loss = cfg["skip_lm_loss"]
    loss_scale = cfg["kd_loss_scale"]

    criterion = {}
    if student_cfg.pipeline_model_parallel_size == 1 or parallel_state.is_pipeline_last_stage():
        criterion[tuple(logit_pair)] = LogitsKLLoss(student_cfg)
        # NOTE: Projection layer shared among intermediate layer pairs.
        projection_layer = ProjectionLayer(student_cfg, teacher_cfg)

        for student_layer, teacher_layer in intermediate_pairs:
            if parallel_state.get_tensor_and_context_parallel_rank() == 0:
                print(
                    "Distillation: Adding intermediate loss between"
                    f" `{student_layer}` of student (hidden size {student_cfg.hidden_size}) and"
                    f" `{teacher_layer}` of teacher (hidden size {teacher_cfg.hidden_size})."
                )
            student_layer = _adjust_layer_index_for_pp(student_layer, student_cfg)
            teacher_layer = _adjust_layer_index_for_pp(teacher_layer, teacher_cfg)
            criterion[(student_layer, teacher_layer)] = HiddenStateCosineLoss(
                student_cfg, projection_layer=projection_layer
            )

    loss_balancer = LogitsAndIntermediatesLossBalancer(kd_loss_scale=loss_scale, skip_original_loss=skip_lm_loss)

    cfg["criterion"] = criterion
    cfg["loss_balancer"] = loss_balancer

    return cfg


def _adjust_layer_index_for_pp(submodule_name, model_cfg):
    """Adjust any sequence-based layer indices found in a submodule name for Pipeline Parallelism."""

    match = re.search(r'(?<=\.)\d+(?=\.)', submodule_name)
    if not match:
        return submodule_name

    offset = TransformerLayer._get_layer_offset(model_cfg)
    new_layer_idx = int(match.group(0)) - offset
    if new_layer_idx < 0:
        raise ValueError(f"Layer {submodule_name} does not fall on final PP rank.")

    new_submodule_name = submodule_name.replace(match.group(0), str(new_layer_idx))
    if parallel_state.get_tensor_and_context_parallel_rank() == 0:
        print(f'Distillation: Renamed layer "{submodule_name}" on final PP rank to "{new_submodule_name}"')
    return new_submodule_name


def teacher_provider(
    config: llm.GPTConfig, ckpt_path: str, tokenizer: "TokenizerSpec", trainer: nl.Trainer
) -> "MCoreGPTModel":
    """Teacher model factory (must be a non-local function to pickle)."""
    logging.info("Distillation: Loading teacher weights...")

    # TODO(aanoosheh): Replace spec with modelopt one
    model = config.configure_model(tokenizer)

    sharded_state_dict = {"state_dict": model.sharded_state_dict(prefix="module.")}
    strict = trainer.strategy.ckpt_load_strictness
    checkpoint = trainer.strategy.checkpoint_io.load_checkpoint(ckpt_path, sharded_state_dict, strict=strict)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}

    # convert from StrictHandling to bool for PTL
    if strict is not None and not isinstance(strict, bool):
        strict = parse_strict_flag(strict)
        strict_options = [
            StrictHandling.ASSUME_OK_UNEXPECTED,
            StrictHandling.RAISE_UNEXPECTED,
            StrictHandling.RAISE_ALL,
        ]
        strict = strict in strict_options
    model.load_state_dict(state_dict, strict=strict)

    torch.cuda.empty_cache()
    logging.info("Distillation: teacher weights loaded.")
    return model


class LoopingCachedDataIterator:
    """Iterator which takes in a sequence and cycles through it when exhausted."""

    def __init__(self, data):
        self.data = data
        self.it = iter(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.data)
            return next(self.it)


def adjust_distillation_model_for_mcore(
    model: DistillationModel, model_cfg: "TransformerConfig", distill_cfg: Dict[str, Any]
):
    """Extra modifications to ``mtd.DistillationModel`` required for Megatron-Core."""
    # Get rid of ModelOpt Distillation state
    # NOTE: If re-placed, above losses need modifcation as `TransformerConfig` has non-pickleable elements.
    existing_state = mto.ModeloptStateManager(model).state_dict()
    assert len(existing_state) == 1 and existing_state[0][0] == "kd_loss", f"{existing_state=}"
    # mto.ModeloptStateManager.remove_state(model)
    delattr(model, mto.ModeloptStateManager._state_key)  # Use above method from modelopt 0.27

    # Hide teacher during `sharded_state_dict` method.
    def _sharded_state_dict(self, *args, **kwargs) -> "ShardedStateDict":
        with self.hide_teacher_model():
            return self._sharded_state_dict(*args, **kwargs)

    model._sharded_state_dict = model.sharded_state_dict
    model.sharded_state_dict = MethodType(_sharded_state_dict, model)

    # Skip `lm_loss` bypassing it when training if not needed for backprop.
    def _compute_language_model_loss(self, labels, logits) -> torch.Tensor:
        if self.training:
            return torch.zeros_like(labels, dtype=logits.dtype)
        return self._compute_language_model_loss(labels, logits)

    if distill_cfg["skip_lm_loss"]:
        model._compute_language_model_loss = model.compute_language_model_loss
        model.compute_language_model_loss = MethodType(_compute_language_model_loss, model)

    # Skip `lm_loss` always for teacher.
    def _compute_language_model_loss(self, labels, logits) -> torch.Tensor:
        return torch.zeros_like(labels, dtype=logits.dtype)

    model.teacher_model.compute_language_model_loss = MethodType(_compute_language_model_loss, model.teacher_model)

    if model_cfg.pipeline_model_parallel_size > 1:

        def _set_input_tensor(self, input_tensor: torch.Tensor):
            obj = self.teacher_model if self._only_teacher_fwd else self
            return type(self).set_input_tensor(obj, input_tensor)

        # Pipeline-parallel Distillation requires a way to cache input batches for subsequent
        # forward calls, as well as a way to pass through output tensors to teacher model.
        model.set_input_tensor = MethodType(_set_input_tensor, model)

        @contextmanager
        def _swap_teacher_config(self, model_wrapper):
            try:
                if hasattr(model_wrapper, "config"):
                    model_wrapper._config = model_wrapper.config
                model_wrapper.config = self.teacher_model.config
                yield
            finally:
                del model_wrapper.config
                if hasattr(model_wrapper, "_config"):
                    model_wrapper.config = model_wrapper._config
                    del model_wrapper._config

        # Pipeline-parallel forward function relies on the config in the model to know what
        # hidden size of tensor to communicate to next stage.
        model.swap_teacher_config = MethodType(_swap_teacher_config, model)
