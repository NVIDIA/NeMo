"""
base_llm.py

Abstract class definition of a large (autoregressive) language model backbone (LLM), with full annotations of class
methods, utility functions, and initialization logic.

We also define the generic HFLLMBackbone class here, providing a default interface for loading any HF
AutoModelForCausalLM (e.g., LLamaForCausalLM). In general, we make the assumption that any given LLM backbone implements
the AutoModelForCausalLM API (though we may add Seq2Seq models in the future).

We make this assumption to keep the LLM handling in this codebase relatively lightweight, and to inherit all the nice HF
utilities around different types of decoding/generation strategies.
"""

import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List, Optional, Sequence, Type

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo.collections.vlm.openvla.data.prismatic.models.backbones.llm.prompting import PromptBuilder
from nemo.collections.vlm.openvla.data.prismatic.overwatch import initialize_overwatch

# Suppress HF Deprecation Warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for arbitrary HF LLM Backbones ===
class LLMBackbone(nn.Module, ABC):
    def __init__(self, llm_backbone_id: str) -> None:
        super().__init__()
        self.identifier = llm_backbone_id

        # Instance attributes for an LLM Backbone
        self.tokenizer: PreTrainedTokenizerBase = None

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer

    @property
    @abstractmethod
    def prompt_builder_fn(self) -> Type[PromptBuilder]: ...

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id


# === Abstract Base Class for Arbitrary HF Causal LLMs ===
class HFCausalLLMBackbone(LLMBackbone, ABC):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_family: str,
        llm_cls: Type[PreTrainedModel],
        hf_hub_path: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = False,
    ) -> None:
        super().__init__(llm_backbone_id)
        self.llm_family = llm_family
        self.llm_max_length = llm_max_length
        self.inference_mode = inference_mode

        # Load (Fast) Tokenizer
        overwatch.info(f"Loading [bold]{llm_family}[/] (Fast) Tokenizer via the AutoTokenizer API", ctx_level=1)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_hub_path, model_max_length=self.llm_max_length, token=hf_token, padding_side="right"
        )

        # Validation =>> Our VLM logic currently operates under the assumption that the tokenization of a new input
        #                starts with a <BOS> token unless `add_special_tokens = False`; for these models, we empirically
        #                find that adding image patches *after* the BOS leads to much better performance.
        #
        # As a result we explicitly validate that a tokenizer conforms to the expected behavior; if you're reading this
        # line, it's probably because you're adding a new LLM with a different tokenizer behavior. If so, feel free to
        # override the `SPECIAL_CASES` set below, but make sure to make the appropriate changes in the `datasets.py`
        # and VLM `forward()` logic!
        SPECIAL_CASES = {
            # Phi-2 Tokenizer doesn't add any BOS tokens by default, and sets BOS == EOS == "<|endoftext|>"
            #   =>> We'll prepend BOS to first input (to play nicely with image token insertion logic; verified that
            #       this works well with base LLM generation.
            #   =>> Like Llama-2 Tokenizers -- we'll add a special PAD token for training purposes.
            "phi-2-3b",
        }
        if self.identifier in SPECIAL_CASES:
            return

        # Note =>> this assert should hold for all Llama-derived tokenizers (`LlamaTokenizerFast` ==> includes Mistral!
        assert (self.tokenizer("Test 123", add_special_tokens=True).input_ids[0] == self.tokenizer.bos_token_id) and (
            self.tokenizer("Test 123", add_special_tokens=False).input_ids[0] != self.tokenizer.bos_token_id
        ), (
            f"Default Tokenizer of type `{type(self.tokenizer)}` does not automatically prefix inputs with BOS token!\n"
            "Please read the comment in `base_llm.py` for more information!"
        )
