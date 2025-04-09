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
import re
import warnings
from itertools import groupby

import torch
import torch.utils.data
from lhotse import CutSet, Seconds, compute_num_frames
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_audio, collate_vectors
from lhotse.utils import ifnone

from nemo.collections.common.data.lhotse import NeMoMultimodalConversation
from nemo.collections.common.data.lhotse.text_adapters import TextTurn
from nemo.collections.common.data.prompt_fn import registered_prompt_format_fn
from nemo.collections.common.prompts import Llama2PromptFormatter
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.utils import logging


class SALMDataset(torch.utils.data.Dataset):
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __getitem__(self, conversations: CutSet) -> dict:
        all_cuts = []
        example_idx_to_audio_idxs = []
        cntr = 0
        for conversation in conversations:
            assert isinstance(conversation, NeMoMultimodalConversation)
            example_idx_to_audio_idxs.append([])
            for cut in conversation.list_cuts():
                all_cuts.append(cut)
                example_idx_to_audio_idxs[-1].append(cntr)
                cntr += 1
        audios, audio_lens = collate_audio(CutSet(all_cuts))
        return {
            "audios": audios,
            "audio_lens": audio_lens,
            "example_idx_to_audio_idxs": example_idx_to_audio_idxs,
            "input_ids": collate_vectors([c.input_ids for c in conversations], padding_value=self.pad_id),
            "answer_offsets": torch.tensor([len(c.context_ids for c in conversations)], dtype=torch.long),
            # "context_ids": collate_vectors([c.context_ids for c in conversations], padding_value=self.pad_id),
            # "answer_ids": collate_vectors([c.answer_ids for c in conversations], padding_value=self.pad_id),
            "loss_mask": collate_vectors([c.loss_mask for c in conversations], padding_value=0).to(torch.bool),
        }


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
    turns[0]["role"] = "system_and_user"
    turns[0]["slots"]["system"] = example.system_prompt
    return prompt.encode_dialog(turns)
