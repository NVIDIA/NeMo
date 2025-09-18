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

from itertools import groupby
from typing import Iterable, Union

import numpy as np
import torch
import torch.utils.data
from lhotse import CutSet, fastcopy
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.common.data.lhotse import NeMoMultimodalConversation
from nemo.collections.common.data.lhotse.text_adapters import (
    AudioTurn,
    TextTurn,
    collate_conversation_audio_fault_tolerant,
)
from nemo.collections.common.data.prompt_fn import registered_prompt_format_fn
from nemo.collections.common.prompts import Llama2PromptFormatter
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.utils import get_pad_id


class HybridSALMTDTDataset(torch.utils.data.Dataset):
    """
    A dataset for Hybrid SALM-TDT models that separates speech and non-speech data.
    
    This dataset processes multimodal conversations and separates them into:
    1. Speech data: conversations with audio turns (goes through both TDT and SALM heads)
    2. Non-speech data: text-only conversations (goes through SALM head only)
    
    Args:
        salm_tokenizer (AutoTokenizer): SALM tokenizer for language model
        tdt_tokenizer (AutoTokenizer): TDT tokenizer for ASR model
        speech_ratio (float): Ratio of speech data in each batch (default: 0.5)
    """

    def __init__(self, salm_tokenizer: AutoTokenizer, tdt_tokenizer: AutoTokenizer, speech_ratio: float = 0.5, is_speech: bool = True) -> None:
        self.salm_tokenizer = salm_tokenizer
        self.tdt_tokenizer = tdt_tokenizer
        self.salm_pad_id = get_pad_id(salm_tokenizer)
        self.tdt_pad_id = get_pad_id(tdt_tokenizer)
        self.speech_ratio = speech_ratio
        self.is_speech = is_speech

    def __getitem__(self, conversations: CutSet) -> dict | None:
        if not conversations:
            return None
        
        # for i, conv in enumerate(conversations):
        #     print(f"DEBUG Conversation: {conv}")
        #     salm_text = self.salm_tokenizer.ids_to_text(conv.input_ids.tolist())
        #     print(f"DEBUG SALM text (with prompts): {salm_text}")
        
        #     # Use pre-tokenized TDT input_ids from lhotse processing
        #     tdt_tokens = conv.tdt_input_ids
        #     tdt_len = conv.tdt_input_ids_len
        #     print(f"DEBUG TDT tokens: {self.tdt_tokenizer.ids_to_text(tdt_tokens)}")
        #     print(f"DEBUG Using pre-tokenized TDT tokens: {tdt_tokens}")
        #     print(f"DEBUG TDT length: {tdt_len}")
            
        audios, audio_lens, conversations = collate_conversation_audio_fault_tolerant(conversations)    
        salm_input_ids = left_collate_vectors([c.input_ids for c in conversations], padding_value=self.salm_pad_id)
        
        tdt_cuts = [c for c in conversations if hasattr(c, 'tdt_input_ids') and c.tdt_input_ids is not None]
        if len(tdt_cuts) > 0:
            tdt_input_ids = right_collate_vectors([c.tdt_input_ids for c in tdt_cuts], padding_value=0)
            tdt_input_ids_len = torch.tensor([c.tdt_input_ids_len for c in tdt_cuts], device=tdt_input_ids.device, dtype=torch.long)
            
            return {
                "audios": audios,
                "audio_lens": audio_lens,
                "input_ids": salm_input_ids,  # For SALM head
                "tdt_input_ids": tdt_input_ids,    # For TDT head
                "tdt_input_ids_len": tdt_input_ids_len,
                "loss_mask": left_collate_vectors(
                    [getattr(c, "mask", torch.empty(0)) for c in conversations], padding_value=0
                ).to(torch.bool),
                "conversations": drop_in_memory_data(conversations),
            }
        else:
            return {
                "audios": audios,
                "audio_lens": audio_lens,
                "input_ids": salm_input_ids,  # For SALM head
                "loss_mask": left_collate_vectors(
                    [getattr(c, "mask", torch.empty(0)) for c in conversations], padding_value=0
                ).to(torch.bool),
                "conversations": drop_in_memory_data(conversations),
            }

    def _extract_conversation_text(self, conversation) -> str:
        """Extract raw text content from a conversation for TDT tokenization (without prompts)."""
        text_parts = []
        for turn in conversation.turns:
            if hasattr(turn, 'value'):  # TextTurn
                # For TDT, we want the raw text content without any prompt formatting
                text_parts.append(turn.value)
            elif hasattr(turn, 'audio_locator_tag'):  # AudioTurn
                # For audio turns in TDT training, we want the transcription text
                if hasattr(turn, 'text') and turn.text:
                    # Use the transcription text from the AudioTurn
                    text_parts.append(turn.text)
                else:
                    # Fallback to audio_locator_tag if no transcription text available
                    # This might need adjustment based on your specific data format
                    text_parts.append(turn.audio_locator_tag)
        return " ".join(text_parts)


def left_collate_vectors(
    tensors: Iterable[Union[torch.Tensor, np.ndarray]],
    padding_value: Union[int, float] = CrossEntropyLoss().ignore_index,
) -> torch.Tensor:
    tensors = [torch.as_tensor(t) for t in tensors]
    assert all(len(t.shape) == 1 for t in tensors), "Expected only 1-D input tensors."
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value, padding_side="left")


def right_collate_vectors(
    tensors: Iterable[Union[torch.Tensor, np.ndarray]],
    padding_value: Union[int, float] = CrossEntropyLoss().ignore_index,
) -> torch.Tensor:
    tensors = [torch.as_tensor(t) for t in tensors]
    assert all(len(t.shape) == 1 for t in tensors), "Expected only 1-D input tensors."
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value, padding_side="right")

def drop_in_memory_data(conversations: CutSet) -> CutSet:
    def _drop(conversation: NeMoMultimodalConversation) -> NeMoMultimodalConversation:
        turns = []
        for t in conversation.turns:
            if isinstance(t, AudioTurn):
                t = fastcopy(t, cut=t.cut.drop_in_memory_data())
            turns.append(t)
        return fastcopy(conversation, turns=turns)

    return conversations.map(_drop, apply_fn=None)


@registered_prompt_format_fn(NeMoMultimodalConversation, Llama2PromptFormatter)
def default_multimodal_conversation_prompt_format_fn(
    example: NeMoMultimodalConversation, prompt: Llama2PromptFormatter
):
    # Collapse consecutive same-role turns into single turn for proper prompt formatting.
    turns = groupby(
        [
            {
                "role": turn.role,
                "slots": {"message": turn.value if isinstance(turn, TextTurn) else turn.audio_locator_tag},
            }
            for turn in example.turns
        ],
        key=lambda turn: turn["role"],
    )
    turns = [
        {"role": role, "slots": {"message": " ".join(t["slots"]["message"] for t in turn_grp)}}
        for role, turn_grp in turns
    ]
    if hasattr(example, "system_prompt"):
        turns[0]["role"] = "system_and_user"
        turns[0]["slots"]["system"] = example.system_prompt
    return prompt.encode_dialog(turns)
