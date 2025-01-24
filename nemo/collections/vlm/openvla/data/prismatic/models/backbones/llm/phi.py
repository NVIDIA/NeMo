"""
phi.py

Class definition for all LLMs derived from PhiForCausalLM.
"""

from typing import Optional, Type

import torch
from torch import nn as nn
from transformers import PhiForCausalLM
from transformers.models.phi.modeling_phi import PhiDecoderLayer

from nemo.collections.vlm.openvla.data.prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from nemo.collections.vlm.openvla.data.prismatic.models.backbones.llm.prompting import PhiPromptBuilder, PromptBuilder

# Registry ==> Support Phi Models (from HF Transformers)
# fmt: off
PHI_MODELS = {
    # === Phi-2 ===
    "phi-2-3b": {
        "llm_family": "phi", "llm_cls": PhiForCausalLM, "hf_hub_path": "microsoft/phi-2"
    }
}
# fmt: on


class PhiLLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **PHI_MODELS[llm_backbone_id],
        )

        # [Special Case] Phi PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        if self.identifier.startswith("phi-2"):
            return PhiPromptBuilder

        raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")