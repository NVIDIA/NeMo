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
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM

from nemo.utils.model_utils import import_class_by_path


def load_pretrained_nemo(cls, model_path_or_name: str, pretrained_weights: bool = True):
    """
    Load pretrained NeMo 1.0 model (inheriting from ModelPT). Works with ASR, TTS, codec models.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.
    """
    if Path(model_path_or_name).exists() and model_path_or_name.endswith(".nemo"):
        # if pretrained_weights:
        return cls.restore_from(model_path_or_name)
    # else:
    #     config = cls.restore_from(model_path_or_name, return_config=True)
    #     return _instantiate_nemo_from_config(cls, config)
    else:
        # if pretrained_weights:
        return cls.from_pretrained(model_path_or_name)
    # else:
    #     config = cls.from_pretrained(model_path_or_name, return_config=True)
    #     return _instantiate_nemo_from_config(cls, config)


def _instantiate_nemo_from_config(cls, config):
    """
    Return a randomly initialized NeMo 1.0 model with architecture corresponding to ``config``.
    Implements some shenaningans to make models load correctly.
    """
    raise NotImplementedError("NeMo 1.0 doesn't support loading model from config without weights reliably.")
    for k in ("train_ds", "validation_ds", "test_ds"):
        if k in config:
            config[k]["defer_setup"] = True  # avoid instantiating datasets and dataloaders
    for k in ("target", "_target_"):
        if k in config:
            cls = import_class_by_path(config[k])  # specialize cls e.g. ASRModel -> EncDecCTCModelBPE
            break
    cwd = os.getcwd()
    with TemporaryDirectory() as tmpdir:
        try:  # hack ported from SaveRestoreConnector to avoid tokenizer loading issue in ASR models
            os.chdir(tmpdir)
            cls._set_model_restore_state(is_being_restored=True, folder=tmpdir)
            return cls.from_config_dict(config)
        finally:
            os.chdir(cwd)


def load_pretrained_hf(model_path_or_name: str, pretrained_weights: bool = True, dtype=torch.float32):
    """
    Load pretrained HuggingFace AutoModelForCausalLM.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.
    """
    if pretrained_weights:
        return AutoModelForCausalLM.from_pretrained(model_path_or_name, torch_dtype=dtype)
    else:
        config = AutoConfig.from_pretrained(model_path_or_name)
        return AutoModelForCausalLM.from_config(config, torch_dtype=dtype)


@contextmanager
def move_embedding(model):
    """Temporarily restores the embedding layer into HF LLM. Supports LoRA models."""
    if isinstance(model.llm, PeftModel):
        model.llm.base_model.model.model.embed_tokens = model.embed_tokens
    else:
        model.llm.model.embed_tokens = model.embed_tokens
    yield
    if isinstance(model.llm, PeftModel):
        del model.llm.base_model.model.model.embed_tokens
    else:
        del model.llm.model.embed_tokens
