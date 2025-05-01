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
Central import hub with graceful‑failure semantics.
"""

import logging
from importlib import import_module
from types import ModuleType
from typing import Any, Dict, List, Tuple

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# Helper utilities                                                             #
# ---------------------------------------------------------------------------- #
def safe_import_from(module: str, name: str | None = None) -> Tuple[Any, bool]:
    """
    Import a symbol or an entire module without crashing the app.

    Parameters
    ----------
    module : str
        Full module path (e.g. 'nemo.collections.llm.gpt.model').
    name : str | None
        If given, the attribute to fetch from the imported module.
        If None, the module itself is returned.

    Returns
    -------
    (object_or_None, bool_success)
    """
    try:
        mod: ModuleType = import_module(module)
        if name is None:
            return mod, True
        return getattr(mod, name), True
    except Exception as exc:  # pragma: no cover
        LOG.debug("Optional import failed: %s%s – %s", module, f'.{name}' if name else '', exc)
        return None, False


# ---------------------------------------------------------------------------- #
# Import registry                                                              #
# ---------------------------------------------------------------------------- #
# Symbol lists taken from the original static import section.
# Keep everything in one dict so it’s easy to extend.
_imports: Dict[str, List[str | None]] = {
    # plain module import -----------------------------------------------------
    "nemo.collections.llm.peft": [None],  # exported as `peft`
    # -------------- data modules --------------------------------------------
    "nemo.collections.llm.gpt.data": [
        "AlpacaDataModule",
        "ChatDataModule",
        "CustomRetrievalDataModule",
        "DollyDataModule",
        "FineTuningDataModule",
        "HFDatasetDataModule",
        "HFDatasetDataModulePacked",
        "HFMockDataModule",
        "MockDataModule",
        "PreTrainingDataModule",
        "SquadDataModule",
    ],
    "nemo.collections.llm.gpt.data.api": ["dolly", "hf_dataset", "mock", "squad"],
    # -------------- GPT models ----------------------------------------------
    "nemo.collections.llm.gpt.model": [
        "Baichuan2Config",
        "Baichuan2Config7B",
        "Baichuan2Model",
        "BaseMambaConfig1_3B",
        "BaseMambaConfig2_7B",
        "BaseMambaConfig130M",
        "BaseMambaConfig370M",
        "BaseMambaConfig780M",
        "ChatGLM2Config6B",
        "ChatGLM3Config6B",
        "ChatGLMConfig",
        "ChatGLMModel",
        "CodeGemmaConfig2B",
        "CodeGemmaConfig7B",
        "CodeLlamaConfig7B",
        "CodeLlamaConfig13B",
        "CodeLlamaConfig34B",
        "CodeLlamaConfig70B",
        "DeepSeekModel",
        "DeepSeekV2Config",
        "DeepSeekV2LiteConfig",
        "DeepSeekV3Config",
        "Gemma2Config",
        "Gemma2Config2B",
        "Gemma2Config9B",
        "Gemma2Config27B",
        "Gemma2Model",
        "GemmaConfig",
        "GemmaConfig2B",
        "GemmaConfig7B",
        "GemmaModel",
        "GPTConfig",
        "GPTConfig5B",
        "GPTConfig7B",
        "GPTConfig20B",
        "GPTConfig40B",
        "GPTConfig126M",
        "GPTConfig175B",
        "GPTModel",
        "HFAutoModelForCausalLM",
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
        "Llama31Nemotron70BConfig",
        "Llama31NemotronNano8BConfig",
        "Llama31NemotronUltra253BConfig",
        "Llama32Config1B",
        "Llama32Config3B",
        "Llama33NemotronSuper49BConfig",
        "LlamaConfig",
        "LlamaModel",
        "LlamaNemotronModel",
        "Llama32EmbeddingConfig1B",
        "Llama32EmbeddingConfig3B",
        "LlamaEmbeddingModel",
        "MambaModel",
        "MaskedTokenLossReduction",
        "MistralConfig7B",
        "MistralModel",
        "MistralNeMoConfig12B",
        "MixtralConfig",
        "MixtralConfig8x3B",
        "MixtralConfig8x7B",
        "MixtralConfig8x22B",
        "MixtralModel",
        "Nemotron3Config4B",
        "Nemotron3Config8B",
        "Nemotron3Config22B",
        "Nemotron4Config15B",
        "Nemotron4Config340B",
        "NemotronConfig",
        "NemotronHConfig8B",
        "NemotronHConfig47B",
        "NemotronHConfig56B",
        "NemotronModel",
        "NVIDIAMambaConfig8B",
        "NVIDIAMambaHybridConfig8B",
        "Phi3Config",
        "Phi3ConfigMini",
        "Phi3Model",
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
        "SSMConfig",
        "Starcoder2Config",
        "Starcoder2Config3B",
        "Starcoder2Config7B",
        "Starcoder2Config15B",
        "Starcoder2Model",
        "StarcoderConfig",
        "StarcoderConfig15B",
        "StarcoderModel",
        "gpt_data_step",
        "gpt_forward_step",
    ],
    # -------------- T5 -------------------------------------------------------
    "nemo.collections.llm.t5.data": [
        ("FineTuningDataModule", "T5FineTuningDataModule"),  # rename later
        ("MockDataModule", "T5MockDataModule"),
        ("PreTrainingDataModule", "T5PreTrainingDataModule"),
        ("SquadDataModule", "T5SquadDataModule"),
    ],
    "nemo.collections.llm.t5.model": [
        "T5Config",
        "T5Config3B",
        "T5Config11B",
        "T5Config220M",
        "T5Model",
        "t5_data_step",
        "t5_forward_step",
    ],
    # -------------- BERT -----------------------------------------------------
    "nemo.collections.llm.bert.data": [
        "BERTMockDataModule",
        "BERTPreTrainingDataModule",
        "SpecterDataModule",
    ],
    "nemo.collections.llm.bert.model": [
        "BertConfig",
        "BertEmbeddingLargeConfig",
        "BertEmbeddingMiniConfig",
        "BertEmbeddingModel",
        "BertModel",
        "HuggingFaceBertBaseConfig",
        "HuggingFaceBertConfig",
        "HuggingFaceBertLargeConfig",
        "HuggingFaceBertModel",
        "MegatronBertBaseConfig",
        "MegatronBertConfig",
        "MegatronBertLargeConfig",
    ],
    # -------------- other loose symbols -------------------------------------
    "nemo.collections": ["tokenizer"],  # may or may not exist
}


# ---------------------------------------------------------------------------- #
# Perform the imports                                                          #
# ---------------------------------------------------------------------------- #
globals_ns = globals()
__all__: List[str] = []

for mod, sym_list in _imports.items():
    for sym in sym_list:
        export_name = None
        target_attr = None

        # Handle (attr, export_alias) tuples (used once for T5*)
        if isinstance(sym, tuple):
            target_attr, export_name = sym
        elif sym is None:  # plain module import
            target_attr = None
            export_name = mod.split(".")[-1]  # last part (e.g. 'peft')
        else:
            target_attr = sym
            export_name = sym

        obj, ok = safe_import_from(mod, target_attr) if target_attr or target_attr is None else (None, False)
        globals_ns[export_name] = obj
        if ok:
            __all__.append(export_name)

# ---------------------------------------------------------------------------- #
# Dynamic import of runtime helpers (train, generate …)                        #
# ---------------------------------------------------------------------------- #
# Keep the original logic but protect imports.
try:
    import nemo_run as run  # noqa: F401

    _api_mod, _ = safe_import_from("nemo.collections.llm.api", None)
    if _api_mod is not None:
        for _name in [
            "distill",
            "export_ckpt",
            "finetune",
            "generate",
            "import_ckpt",
            "pretrain",
            "prune",
            "ptq",
            "train",
            "validate",
        ]:
            globals_ns[_name], ok = safe_import_from("nemo.collections.llm.api", _name)
            if ok:
                __all__.append(_name)

        # recipes: `from nemo.collections.llm.recipes import *`
        _recipes_mod, _ = safe_import_from("nemo.collections.llm.recipes", None)
        if _recipes_mod is not None:
            globals_ns.update(_recipes_mod.__dict__)
except ImportError as exc:
    LOG.warning("Failed to import nemo_run / LLM api helpers: %s", exc)

# deploy / evaluate are optional too
for _single in ("deploy", "evaluate"):
    obj, ok = safe_import_from("nemo.collections.llm.api", _single)
    if ok:
        globals_ns[_single] = obj
        __all__.append(_single)

# ---------------------------------------------------------------------------- #
# Final clean‑up / namespace polish                                            #
# ---------------------------------------------------------------------------- #
del _imports, sym_list, sym, export_name, target_attr, ok, obj, _single, _name
