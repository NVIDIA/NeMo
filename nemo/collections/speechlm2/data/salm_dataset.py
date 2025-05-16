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

import torch
import torch.utils.data
from lhotse import CutSet, fastcopy
from lhotse.dataset.collation import collate_audio, collate_vectors

from nemo.collections.common.data.lhotse import NeMoMultimodalConversation
from nemo.collections.common.data.lhotse.text_adapters import AudioTurn, TextTurn
from nemo.collections.common.data.prompt_fn import registered_prompt_format_fn
from nemo.collections.common.prompts import Llama2PromptFormatter, Llama3PromptFormatter, PromptFormatter
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.utils import get_pad_id


class SALMDataset(torch.utils.data.Dataset):
    """
    A dataset for Speech-Augmented Language Models (SALM) that processes multimodal conversations
    containing both text and audio turns.

    This dataset handles NeMoMultimodalConversation objects which combine text messages
    and audio segments in a conversational format. It uses audio_locator_tag in the text,
    where each such placeholder corresponds to an entire audio segment.

    Args:
        tokenizer (AutoTokenizer):
            Tokenizer for converting text to token IDs and vice versa. Must have a special
            audio_locator_tag token that will be replaced with audio embeddings during model's
            training step.

    Returns:
        A dictionary with the following keys:
            - audios: Tensor of audio waveform samples [B_audio, T_samples]
            - audio_lens: Tensor of audio lengths [B_audio]
            - input_ids: Tensor of text token IDs [B, T_tokens], including audio_locator_tag tokens
            - loss_mask: Boolean tensor [B, T_tokens] indicating which tokens are part of the
                assistant's responses (True) and should be used for computing loss

    Notes:
        - Each audio_locator_tag token in input_ids corresponds to an audio segment in audios
        - The SALM model later replaces these audio_locator_tag tokens with encoded audio embeddings
        - The loss_mask identifies which tokens are part of the target sequences (assistant responses)
          and which are part of the source sequences (user prompts)
        - The input_ids and loss_mask will be expanded during model forward pass to account for
          the variable-length audio segments that replace each audio_locator_tag token
    """

    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer
        self.pad_id = get_pad_id(tokenizer)

    def __getitem__(self, conversations: CutSet) -> dict:
        all_cuts = []
        for conversation in conversations:
            for cut in conversation.list_cuts():
                all_cuts.append(cut)
        audios, audio_lens = collate_audio(CutSet(all_cuts))
        audio_conv = conversations.map(drop_response, apply_fn=None).to_eager()
        text_conv = audio_conv.map(to_text_conversation, apply_fn=None).to_eager()
        return {
            "audios": audios,
            "audio_lens": audio_lens,
            "audio_conversations": audio_conv,
            "text_conversations": text_conv,
        }


def drop_response(conv: NeMoMultimodalConversation) -> NeMoMultimodalConversation:
    if conv.turns[-1].role == "assistant":
        return fastcopy(conv, turns=conv.turns[:-1], custom=conv.custom.copy())


def to_text_conversation(conv: NeMoMultimodalConversation) -> NeMoMultimodalConversation:
    turns = []
    for turn in conv.turns:
        if isinstance(turn, AudioTurn):
            assert len(turn.cut.supervisions) > 0, f"The cut has no supervisions. {turn.cut=}"
            assert turn.cut.supervisions[0].text is not None, f"The cut's supervision has no text. {turn.cut=}"
            turn = TextTurn(value=turn.cut.supervisions[0].text, role=turn.role)
        turns.append(turn)
    return NeMoMultimodalConversation(
        id=conv.id,
        turns=turns,
        token_equivalent_duration=conv.token_equivalent_duration,
        custom=conv.custom.copy(),
    )


@registered_prompt_format_fn(NeMoMultimodalConversation, Llama3PromptFormatter)
def default_multimodal_conversation_prompt_format_fn(
    example: NeMoMultimodalConversation, prompt: Llama3PromptFormatter
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
        turns = [{"role": "system", "slots": {"message": example.system_prompt}}] + turns
    return prompt.encode_dialog(turns)


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
