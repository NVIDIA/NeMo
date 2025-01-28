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

from nemo.collections.llm.gpt.model.baichuan import Baichuan2Config, Baichuan2Config7B, Baichuan2Model
from nemo.collections.llm.gpt.model.base import (
    GPTConfig,
    GPTConfig5B,
    GPTConfig7B,
    GPTConfig20B,
    GPTConfig40B,
    GPTConfig126M,
    GPTConfig175B,
    GPTModel,
    MaskedTokenLossReduction,
    gpt_data_step,
    gpt_forward_step,
    local_layer_spec,
    transformer_engine_full_layer_spec,
    transformer_engine_layer_spec,
)
from nemo.collections.llm.gpt.model.chatglm import ChatGLM2Config6B, ChatGLM3Config6B, ChatGLMConfig, ChatGLMModel
from nemo.collections.llm.gpt.model.deepseek import DeepSeekModel, DeepSeekV2Config, DeepSeekV3Config
from nemo.collections.llm.gpt.model.gemma import (
    CodeGemmaConfig2B,
    CodeGemmaConfig7B,
    GemmaConfig,
    GemmaConfig2B,
    GemmaConfig7B,
    GemmaModel,
)
from nemo.collections.llm.gpt.model.gemma2 import (
    Gemma2Config,
    Gemma2Config2B,
    Gemma2Config9B,
    Gemma2Config27B,
    Gemma2Model,
)
from nemo.collections.llm.gpt.model.hf_auto_model_for_causal_lm import HFAutoModelForCausalLM
from nemo.collections.llm.gpt.model.llama import (
    CodeLlamaConfig7B,
    CodeLlamaConfig13B,
    CodeLlamaConfig34B,
    CodeLlamaConfig70B,
    Llama2Config7B,
    Llama2Config13B,
    Llama2Config70B,
    Llama3Config8B,
    Llama3Config70B,
    Llama31Config8B,
    Llama31Config70B,
    Llama31Config405B,
    Llama32Config1B,
    Llama32Config3B,
    LlamaConfig,
    LlamaModel,
)
from nemo.collections.llm.gpt.model.mistral import MistralConfig7B, MistralModel, MistralNeMoConfig12B
from nemo.collections.llm.gpt.model.mixtral import (
    MixtralConfig,
    MixtralConfig8x3B,
    MixtralConfig8x7B,
    MixtralConfig8x22B,
    MixtralModel,
)
from nemo.collections.llm.gpt.model.nemotron import (
    Nemotron3Config4B,
    Nemotron3Config8B,
    Nemotron3Config22B,
    Nemotron4Config15B,
    Nemotron4Config340B,
    NemotronConfig,
    NemotronModel,
)
from nemo.collections.llm.gpt.model.phi3mini import Phi3Config, Phi3ConfigMini, Phi3Model
from nemo.collections.llm.gpt.model.qwen2 import (
    Qwen2Config,
    Qwen2Config1P5B,
    Qwen2Config7B,
    Qwen2Config72B,
    Qwen2Config500M,
    Qwen2Model,
)
from nemo.collections.llm.gpt.model.ssm import (
    BaseMambaConfig1_3B,
    BaseMambaConfig2_7B,
    BaseMambaConfig130M,
    BaseMambaConfig370M,
    BaseMambaConfig780M,
    NVIDIAMambaConfig8B,
    NVIDIAMambaHybridConfig8B,
    SSMConfig,
)
from nemo.collections.llm.gpt.model.starcoder import StarcoderConfig, StarcoderConfig15B, StarcoderModel
from nemo.collections.llm.gpt.model.starcoder2 import (
    Starcoder2Config,
    Starcoder2Config3B,
    Starcoder2Config7B,
    Starcoder2Config15B,
    Starcoder2Model,
)

__all__ = [
    "Baichuan2Config",
    "Baichuan2Config7B",
    "Baichuan2Model",
    "BaseMambaConfig130M",
    "BaseMambaConfig1_3B",
    "BaseMambaConfig2_7B",
    "BaseMambaConfig370M",
    "BaseMambaConfig780M",
    "ChatGLM2Config6B",
    "ChatGLM3Config6B",
    "ChatGLMConfig",
    "ChatGLMModel",
    "CodeGemmaConfig2B",
    "CodeGemmaConfig7B",
    "CodeLlamaConfig13B",
    "CodeLlamaConfig34B",
    "CodeLlamaConfig70B",
    "CodeLlamaConfig7B",
    "DeepSeekModel",
    "DeepSeekV2Config",
    "DeepSeekV3Config",
    "GPTConfig",
    "GPTModel",
    "Gemma2Config",
    "Gemma2Config27B",
    "Gemma2Config2B",
    "Gemma2Config9B",
    "Gemma2Model",
    "GemmaConfig",
    "GemmaConfig2B",
    "GemmaConfig7B",
    "GemmaModel",
    "HFAutoModelForCausalLM",
    "Llama2Config13B",
    "Llama2Config70B",
    "Llama2Config7B",
    "Llama31Config405B",
    "Llama31Config70B",
    "Llama31Config8B",
    "Llama32Config1B",
    "Llama32Config3B",
    "Llama3Config70B",
    "Llama3Config8B",
    "LlamaConfig",
    "LlamaModel",
    "MaskedTokenLossReduction",
    "MistralConfig7B",
    "MistralModel",
    "MixtralConfig",
    "MixtralConfig8x22B",
    "MixtralConfig8x3B",
    "MixtralConfig8x7B",
    "MixtralModel",
    "NVIDIAMambaConfig8B",
    "NVIDIAMambaHybridConfig8B",
    "Nemotron3Config22B",
    "Nemotron3Config4B",
    "Nemotron3Config8B",
    "Nemotron4Config15B",
    "Nemotron4Config340B",
    "NemotronConfig",
    "NemotronModel",
    "Phi3Config",
    "Phi3ConfigMini",
    "Phi3Model",
    "Qwen2Config",
    "Qwen2Config1P5B",
    "Qwen2Config500M",
    "Qwen2Config72B",
    "Qwen2Config7B",
    "Qwen2Model",
    "SSMConfig",
    "Starcoder2Config",
    "Starcoder2Config15B",
    "Starcoder2Config3B",
    "Starcoder2Config7B",
    "Starcoder2Model",
    "StarcoderConfig",
    "StarcoderConfig15B",
    "StarcoderModel",
    "gpt_data_step",
    "gpt_forward_step",
    "local_layer_spec",
    "transformer_engine_full_layer_spec",
    "transformer_engine_layer_spec",
]
