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

"""
Fault‑tolerant import hub.

Every symbol is imported with `safe_import_from`.  
If the import fails the variable is set to `None` and the app keeps running.
"""

from importlib import import_module
import logging
from types import ModuleType
from typing import Any, Dict, List, Tuple

LOG = logging.getLogger(__name__)

# -----------------------------------------------------------------------------#
# Helper
# -----------------------------------------------------------------------------#
def safe_import_from(module: str, name: str) -> Tuple[Any, bool]:
    """
    Try:  `from <module> import <name>`
    Returns
        (imported_object_or_None, success_bool)
    """
    try:
        mod: ModuleType = import_module(module)
        return getattr(mod, name), True
    except Exception as exc:  # pragma: no cover
        LOG.debug("Optional import failed: %s.%s – %s", module, name, exc)
        return None, False


# -----------------------------------------------------------------------------#
# Special case: Float8Tensor
# -----------------------------------------------------------------------------#
Float8Tensor, HAVE_TE_FLOAT8TENSOR = safe_import_from(
    "transformer_engine.pytorch.float8_tensor", "Float8Tensor"
)


# -----------------------------------------------------------------------------#
# One registry describing every optional symbol
# -----------------------------------------------------------------------------#
_imports: Dict[str, List[str]] = {
    # --------- nemo.collections.llm.gpt.model.baichuan -----------------------
    "nemo.collections.llm.gpt.model.baichuan": [
        "Baichuan2Config",
        "Baichuan2Config7B",
        "Baichuan2Model",
    ],
    # --------- nemo.collections.llm.gpt.model.base ---------------------------
    "nemo.collections.llm.gpt.model.base": [
        "GPTConfig",
        "GPTConfig5B",
        "GPTConfig7B",
        "GPTConfig20B",
        "GPTConfig40B",
        "GPTConfig126M",
        "GPTConfig175B",
        "GPTModel",
        "MaskedTokenLossReduction",
        "gpt_data_step",
        "gpt_forward_step",
        "local_layer_spec",
        "transformer_engine_full_layer_spec",
        "transformer_engine_layer_spec",
    ],
    # --------- nemo.collections.llm.gpt.model.chatglm ------------------------
    "nemo.collections.llm.gpt.model.chatglm": [
        "ChatGLM2Config6B",
        "ChatGLM3Config6B",
        "ChatGLMConfig",
        "ChatGLMModel",
    ],
    # --------- nemo.collections.llm.gpt.model.deepseek -----------------------
    "nemo.collections.llm.gpt.model.deepseek": [
        "DeepSeekModel",
        "DeepSeekV2Config",
        "DeepSeekV2LiteConfig",
        "DeepSeekV3Config",
    ],
    # --------- nemo.collections.llm.gpt.model.gemma --------------------------
    "nemo.collections.llm.gpt.model.gemma": [
        "CodeGemmaConfig2B",
        "CodeGemmaConfig7B",
        "GemmaConfig",
        "GemmaConfig2B",
        "GemmaConfig7B",
        "GemmaModel",
    ],
    # --------- nemo.collections.llm.gpt.model.gemma2 -------------------------
    "nemo.collections.llm.gpt.model.gemma2": [
        "Gemma2Config",
        "Gemma2Config2B",
        "Gemma2Config9B",
        "Gemma2Config27B",
        "Gemma2Model",
    ],
    # --------- nemo.collections.llm.gpt.model.hf_auto_model_for_causal_lm ---
    "nemo.collections.llm.gpt.model.hf_auto_model_for_causal_lm": [
        "HFAutoModelForCausalLM",
    ],
    # --------- nemo.collections.llm.gpt.model.hf_llama_embedding ------------
    "nemo.collections.llm.gpt.model.hf_llama_embedding": [
        "get_llama_bidirectional_hf_model",
    ],
    # --------- nemo.collections.llm.gpt.model.hyena --------------------------
    "nemo.collections.llm.gpt.model.hyena": [
        "Hyena1bConfig",
        "Hyena7bARCLongContextConfig",
        "Hyena7bConfig",
        "Hyena40bARCLongContextConfig",
        "Hyena40bConfig",
        "HyenaConfig",
        "HyenaModel",
        "HyenaNV1bConfig",
        "HyenaNV7bConfig",
        "HyenaNV40bConfig",
        "HyenaNVTestConfig",
        "HyenaTestConfig",
    ],
    # --------- nemo.collections.llm.gpt.model.llama --------------------------
    "nemo.collections.llm.gpt.model.llama": [
        "CodeLlamaConfig7B",
        "CodeLlamaConfig13B",
        "CodeLlamaConfig34B",
        "CodeLlamaConfig70B",
        "Llama2Config7B",
        "Llama2Config13B",
        "Llama2Config70B",
        "Llama3Config8B",
        "Llama3Config70B",
        "Llama4Config",
        "Llama4Experts16Config",
        "Llama4Experts128Config",
        "Llama31Config8B",
        "Llama31Config70B",
        "Llama31Config405B",
        "Llama32Config1B",
        "Llama32Config3B",
        "LlamaConfig",
        "LlamaModel",
        "MLPerfLoRALlamaModel",
    ],
    # --------- nemo.collections.llm.gpt.model.llama_embedding ---------------
    "nemo.collections.llm.gpt.model.llama_embedding": [
        "Llama32EmbeddingConfig1B",
        "Llama32EmbeddingConfig3B",
        "LlamaEmbeddingModel",
    ],
    # --------- nemo.collections.llm.gpt.model.llama_nemotron -----------------
    "nemo.collections.llm.gpt.model.llama_nemotron": [
        "Llama31Nemotron70BConfig",
        "Llama31NemotronNano8BConfig",
        "Llama31NemotronUltra253BConfig",
        "Llama33NemotronSuper49BConfig",
        "LlamaNemotronModel",
    ],
    # --------- nemo.collections.llm.gpt.model.mistral ------------------------
    "nemo.collections.llm.gpt.model.mistral": [
        "MistralConfig7B",
        "MistralModel",
        "MistralNeMoConfig12B",
    ],
    # --------- nemo.collections.llm.gpt.model.mixtral ------------------------
    "nemo.collections.llm.gpt.model.mixtral": [
        "MixtralConfig",
        "MixtralConfig8x3B",
        "MixtralConfig8x7B",
        "MixtralConfig8x22B",
        "MixtralModel",
    ],
    # --------- nemo.collections.llm.gpt.model.nemotron -----------------------
    "nemo.collections.llm.gpt.model.nemotron": [
        "NemotronConfig",
        "Nemotron3Config4B",
        "Nemotron3Config8B",
        "Nemotron3Config22B",
        "Nemotron4Config15B",
        "Nemotron4Config340B",
        "NemotronModel",
    ],
    # --------- nemo.collections.llm.gpt.model.phi3mini -----------------------
    "nemo.collections.llm.gpt.model.phi3mini": [
        "Phi3Config",
        "Phi3ConfigMini",
        "Phi3Model",
    ],
    # --------- nemo.collections.llm.gpt.model.qwen2 --------------------------
    "nemo.collections.llm.gpt.model.qwen2": [
        "Qwen2Config",
        "Qwen2Config1P5B",
        "Qwen2Config7B",
        "Qwen2Config72B",
        "Qwen2Config500M",
        "Qwen2Model",
        "Qwen25Config1P5B",
        "Qwen25Config7B",
        "Qwen25Config14B",
        "Qwen25Config32B",
        "Qwen25Config72B",
        "Qwen25Config500M",
    ],
    # --------- nemo.collections.llm.gpt.model.ssm ---------------------------
    "nemo.collections.llm.gpt.model.ssm": [
        "BaseMambaConfig1_3B",
        "BaseMambaConfig2_7B",
        "BaseMambaConfig130M",
        "BaseMambaConfig370M",
        "BaseMambaConfig780M",
        "MambaModel",
        "NemotronHConfig8B",
        "NemotronHConfig47B",
        "NemotronHConfig56B",
        "NVIDIAMambaConfig8B",
        "NVIDIAMambaHybridConfig8B",
        "SSMConfig",
    ],
    # --------- nemo.collections.llm.gpt.model.starcoder ----------------------
    "nemo.collections.llm.gpt.model.starcoder": [
        "StarcoderConfig",
        "StarcoderConfig15B",
        "StarcoderModel",
    ],
    # --------- nemo.collections.llm.gpt.model.starcoder2 ---------------------
    "nemo.collections.llm.gpt.model.starcoder2": [
        "Starcoder2Config",
        "Starcoder2Config3B",
        "Starcoder2Config7B",
        "Starcoder2Config15B",
        "Starcoder2Model",
    ],
}

# -----------------------------------------------------------------------------#
# Dynamically perform all imports
# -----------------------------------------------------------------------------#
globals_ns = globals()
__all__: List[str] = ["Float8Tensor", "HAVE_TE_FLOAT8TENSOR"]  # always present

for mod, symbols in _imports.items():
    for sym in symbols:
        obj, ok = safe_import_from(mod, sym)
        if ok:
            globals_ns[sym] = obj
            __all__.append(sym)

# ---------------------------------------------------------------------------- #
# Final clean‑up / namespace polish                                            #
# ---------------------------------------------------------------------------- #
del _imports, sym, ok
