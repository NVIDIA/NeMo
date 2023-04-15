# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import get_samples_mapping
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import JSONLMemMapDataset
from nemo.core.classes import Dataset
from nemo.utils import logging
import copy

__all__ = ['GPTSFTChatDataset']

IGNORE_INDEX = -100

CONVERSATION = {
    "system": "A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    "human": "Human",
    "gpt": "Assistant",
}


def _mask_targets(target, tokenized_lens, speakers, header_len, s_ids):
    cur_idx = header_len
    tgt_len = target.shape[0]
    for tokenized_len, speaker, s_id in zip(tokenized_lens, speakers, s_ids):
        if cur_idx >= tgt_len:
            break
        elif cur_idx + tokenized_len < tgt_len:
            # Check whether the mask is applied to the correct position
            if not torch.equal(target[cur_idx + 2:cur_idx + tokenized_len],
                               s_id[2:]):
                logging.warning("a sentence mismatches the corresponding piece "
                                "in the conversation")
        if speaker == "human":
            target[cur_idx:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    unknown_role = "unknown"  # use default unknown role
    roles = {
        "human": CONVERSATION['human'],  # human role
        "gpt": CONVERSATION['gpt'],  # gpt role
    }
    for sentence in source:
        sentence_from = sentence["from"].lower()
        sentence["value"] = (
            BEGIN_SIGNAL
            + roles.get(sentence_from, unknown_role)
            + ": "
            + sentence["value"]
            + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["value"]
    return conversation


def preprocess(
    source: list,
    tokenizer: TokenizerSpec,
):
    """
    Given a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    header = f"{CONVERSATION['system']}\n\n"
    conversation = _add_speaker_and_signal(header, source)
    # tokenize conversations
    input_ids = tokenizer.text_to_ids(conversation)
    target = copy.deepcopy(input_ids)
    header_len = len(tokenizer.text_to_ids(header))

    ids = []
    tokenized_lens = []
    for s in source:
        tokenized_sentence = tokenizer.text_to_ids(s["value"])
        ids.append(torch.tensor(tokenized_sentence))
        tokenized_lens.append(len(tokenized_sentence))
    speakers = [sentence["from"] for sentence in source]
    target = torch.LongTensor(target)
    # not going to train on the header
    target[:header_len] = IGNORE_INDEX
    input_ids = torch.LongTensor(input_ids)

    _mask_targets(target, tokenized_lens, speakers, header_len, ids)
    mask = (target != IGNORE_INDEX).bool()
    assert mask.sum().item() != 0, "mask is empty"
    return dict(input_ids=input_ids, mask=mask)


class GPTSFTChatDataset(GPTSFTDataset):

    def _process_example(self, example):
        """
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.
        """
        result = preprocess(example['conversations'], self.tokenizer)

        return result

    def collate_fn(self, batch):
        input_ids = [item['input_ids'][:-1].tolist() for item in batch]
        labels = [item['input_ids'][1:].tolist() for item in batch]
        loss_mask = [item['mask'][1:].tolist() for item in batch]

        max_length = max([len(x) for x in input_ids])
        if max_length > self.max_seq_length:
            # truncate the sequences if it is longer than max_seq_length
            input_ids = [x[:self.max_seq_length] for x in input_ids]
            labels = [x[:self.max_seq_length] for x in labels]
            loss_mask = [x[:self.max_seq_length] for x in loss_mask]
        # increase max length to nearest multiple of 4 or 8
        if self.pad_to_max_length:
            max_length = self.max_seq_length
        else:
            max_length = min(self.max_seq_length, self._round_to_nearest(max_length, 8))
        assert max_length <= self.max_seq_length

        attention_mask = [self._create_attention_mask(max_length) for _ in batch]
        attention_mask = torch.stack(attention_mask)
        position_ids = [list(range(max_length)) for _ in batch]
        position_ids = torch.LongTensor(position_ids)
        input_ids = torch.LongTensor(
            self._collate_item(input_ids, max_length=max_length, pad_id=self.tokenizer.eos_id)
        )
        labels = torch.LongTensor(self._collate_item(labels, max_length=max_length, pad_id=self.tokenizer.eos_id))
        loss_mask = torch.LongTensor(self._collate_item(loss_mask, max_length=max_length, pad_id=0))

        processed_batch = {
            'tokens': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
        }

        return processed_batch

