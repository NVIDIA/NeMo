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

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import Annotated

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel, torch_dtype_from_mcore_config
from nemo.collections.llm.gpt.model.llama import _export_embedding, _export_head
from nemo.collections.llm.utils import Config
from nemo.lightning import io, teardown
from nemo.lightning.pytorch.optim import OptimizerModule
from nemo.lightning.pytorch.utils import dtype_from_hf

if TYPE_CHECKING:
    from transformers import MistralConfig, MistralForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

HF_TO_MCORE_REGISTRY = {
    'Phi3ForCausalLM': llm.Phi3Model,
    'MistralForCausalLM' llm.MistralModel,
    'GemmaForCausalLM': llm.GemmaModel,
    'Gemma2ForCausalLM': llm.Gemma2Model,
    'LlamaForCausalLM': llm.LlamaModel,
    'BaichuanForCausalLM': llm.Baichuan2Model,
    'Qwen2ForCausalLM': llm.Qwen2Model,
    'Starcoder2ForCausal': llm.Starcoder2Model,
}


class AutoModel(GPTModel):
    def __init__(
        self,
        model: str,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
    from transformers import AutoConfig
    architectures = AutoConfig.from_pretrained(model, trust_remote_code=True).architectures
    assert isinstance(architectures, list), "Expected architectures to be a list"
    assert len(architectures) == 1, "Expected architectures to contain one item"

    if not model in HF_TO_MCORE_REGISTRY:
        raise ValueError("Architecture " + str(architectures) + " not supported")
    model_cls = HF_TO_MCORE_REGISTRY[model]
    self.__class__ = model_cls
    config = model_cls.importer(f'hf://{path}').config
    super().__init__(
        config, optim=optim, tokenizer=tokenizer, model_transform=model_transform
    )

__all__ = [
    "AutoModel",
]
