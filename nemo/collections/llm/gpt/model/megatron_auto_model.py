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

from typing import Callable, Optional

from torch import nn
from nemo.lightning.pytorch.optim import OptimizerModule
from pathlib import Path

from nemo.collections.llm.gpt.model.base import GPTModel
from nemo.collections.llm.gpt.model.chatglm import ChatGLMModel
from nemo.collections.llm.gpt.model.phi3mini import Phi3Model
from nemo.collections.llm.gpt.model.mixtral import MixtralModel
from nemo.collections.llm.gpt.model.mistral import MistralModel
from nemo.collections.llm.gpt.model.gemma import GemmaModel
from nemo.collections.llm.gpt.model.gemma2 import Gemma2Model
from nemo.collections.llm.gpt.model.llama import LlamaModel
from nemo.collections.llm.gpt.model.baichuan import Baichuan2Model
from nemo.collections.llm.gpt.model.qwen2 import Qwen2Model
from nemo.collections.llm.gpt.model.starcoder import StarcoderModel
from nemo.collections.llm.gpt.model.starcoder2 import Starcoder2Model

HF_TO_MCORE_REGISTRY = {
    'ChatGLMForCausalLM': ChatGLMModel,
    'Phi3ForCausalLM': Phi3Model,
    'MixtralForCausalLM': MixtralModel,
    'MistralForCausalLM': MistralModel,
    'GemmaForCausalLM': GemmaModel,
    'Gemma2ForCausalLM': Gemma2Model,
    'LlamaForCausalLM': LlamaModel,
    'BaichuanForCausalLM': Baichuan2Model,
    'Qwen2ForCausalLM': Qwen2Model,
    'StarcoderForCausal': StarcoderModel,
    'Starcoder2ForCausal': Starcoder2Model,
}


class MegatronAutoModel(GPTModel):
    def __init__(
        self,
        model_name_or_path: str,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        """
        An auto-model class which uses an HF model identifier (or model snapshot path) to resolve
        which NeMo class it corresponds to, initializes that class and returns the object, e.g.
        > import nemo.collections.llm as llm
        > m = llm.MegatronAutoModel("microsoft/Phi-3-mini-128k-instruct")
        > print(m)
        >> Phi3Model()

        Type of m is not MegatronAutoModel but instead Phi3Model.
        """

        # Get model class from registry
        from transformers import AutoConfig
        architectures = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True).architectures
        assert isinstance(architectures, list), "Expected architectures to be a list"
        assert len(architectures) == 1, "Expected architectures to contain one item"
        assert isinstance(architectures[0], str), "Expected architecture to be a string"
        if not architectures[0] in HF_TO_MCORE_REGISTRY:
            raise ValueError("Architecture " + str(architectures) + " not supported")
        model_cls = HF_TO_MCORE_REGISTRY[architectures[0]]

        # Get model config using model_cls + corresponding importer
        import_path = model_name_or_path
        if Path(import_path).exists():
            import_path = str(Path(import_path).absolute())
            import_path = f'hf://{import_path}'
        elif not import_path.startswith('hf://'):
            import_path = f'hf://{import_path}'
        config = model_cls.importer(import_path).config

        # Init class
        super().__init__(
            config, optim=optim, tokenizer=tokenizer, model_transform=model_transform
        )

        # Change self's class to model_cls
        self.__class__ = model_cls

__all__ = [
    "MegatronAutoModel",
]
