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

import copy
import random
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Iterator, Literal, Optional, Union

import numpy as np
import torch
from lhotse import Recording
from lhotse.cut import Cut
from lhotse.dataset.dataloading import resolve_seed
from lhotse.serialization import load_jsonl
from lhotse.utils import Pathlike

from nemo.collections.common.data.lhotse.nemo_adapters import expand_sharded_filepaths
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer, TokenizerWrapper
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging

"""
Basic text example, adequate for pretraining-style language modeling.
"""


@dataclass
class TextExample:
    """
    Represents a single text example. Useful e.g. for language modeling.
    """

    text: str
    language: str | None = None
    tokens: Optional[np.ndarray] = None

    @property
    def num_tokens(self) -> Optional[int]:
        if self.tokens is None:
            return None
        return len(self.tokens)

    def tokenize(self, tokenizer: TokenizerWrapper) -> "TextExample":
        self.tokens = np.asarray(tokenizer(self.text, self.language))
        return self


@dataclass
class LhotseTextAdapter:
    """
    ``LhotseTextAdapter`` is used to read a text file and wrap
    each line into a ``TextExample``.
    """

    paths: Union[Pathlike, list[Pathlike]]
    language: str | None = None
    shuffle_shards: bool = False
    shard_seed: Union[int, Literal["trng", "randomized"]] = "trng"

    def __post_init__(self):
        self.paths = expand_sharded_filepaths(self.paths)

    def __iter__(self) -> Iterator[TextExample]:
        paths = self.paths
        if self.shuffle_shards:
            seed = resolve_seed(self.shard_seed)
            random.Random(seed).shuffle(paths)
        for path in paths:
            with open(path) as f:
                for line in f:
                    yield TextExample(line, language=self.language)


"""
Source-target text examples (e.g., machine translation).
"""


@dataclass
class SourceTargetTextExample:
    """
    Represents a pair of text examples. Useful e.g. for sequence-to-sequence tasks.
    Supports a ``question`` field, used as the prompt for LLM.
    """

    source: TextExample
    target: TextExample
    question: TextExample | None = None
    input_ids: np.ndarray | None = None
    context_ids: np.ndarray | None = None
    answer_ids: np.ndarray | None = None
    mask: np.ndarray | None = None

    @property
    def num_tokens(self) -> Optional[int]:
        if self.input_ids is not None:
            return self.input_ids.shape[0]
        return None

    def tokenize(self, tokenizer: TokenizerWrapper) -> "TextExample":
        input_ids = []
        context_ids = []
        if self.question:
            ans = tokenizer(self.question.text, self.question.language)
            input_ids.extend(ans)
            context_ids.extend(ans)
        ans = tokenizer(self.source.text, self.source.language)
        input_ids.extend(ans)
        context_ids.extend(ans)

        answer_ids = tokenizer(self.target.text, self.target.language)
        input_ids.extend(answer_ids)

        self.input_ids = np.asarray(input_ids)
        self.context_ids = np.asarray(context_ids)
        self.answer_ids = np.asarray(answer_ids)
        mask = np.zeros_like(self.input_ids, dtype=np.bool_)
        mask[self.context_ids.shape[0] :] = True
        self.mask = mask

        return self


@dataclass
class LhotseTextPairAdapter:
    """
    ``LhotseTextAdapter`` is used to read a tuple of N text files
    (e.g., a pair of files with translations in different languages)
    and wrap them in a ``TextExample`` object to enable dataloading
    with Lhotse together with training examples in audio modality.

    Provide ``questions_path`` to enable randomly sampling lines with questions.
    """

    source_paths: Union[Pathlike, list[Pathlike]]
    target_paths: Union[Pathlike, list[Pathlike]]
    source_language: str | None = None
    target_language: str | None = None
    questions_path: Pathlike = None
    questions_language: str = None
    shuffle_shards: bool = False
    shard_seed: Union[int, Literal["trng", "randomized"]] = "trng"

    def __post_init__(self):
        ASSERT_MSG = "Both source and target must be a single path or lists of paths"
        if isinstance(self.source_paths, (str, Path)):
            assert isinstance(self.target_paths, (str, Path)), ASSERT_MSG
        else:
            assert isinstance(self.source_paths, list) and isinstance(self.target_paths, list), ASSERT_MSG
            assert len(self.source_paths) == len(
                self.target_paths
            ), f"Source ({len(self.source_paths)}) and target ({len(self.target_paths)}) path lists must have the same number of items."
        self.source_paths = expand_sharded_filepaths(self.source_paths)
        self.target_paths = expand_sharded_filepaths(self.target_paths)

    def __iter__(self) -> Iterator[SourceTargetTextExample]:
        seed = resolve_seed(self.shard_seed)
        rng = random.Random(seed)
        paths = list(zip(self.source_paths, self.target_paths))
        if self.shuffle_shards:
            rng.shuffle(paths)
        questions = None
        if self.questions_path is not None:
            with open(self.questions_path) as f:
                questions = [q.strip() for q in f]
        for source_path, target_path in paths:
            with open(source_path) as fs, open(target_path) as ft:
                for ls, lt in zip(fs, ft):
                    yield SourceTargetTextExample(
                        source=TextExample(ls.strip(), language=self.source_language),
                        target=TextExample(lt.strip(), language=self.target_language),
                        question=(
                            TextExample(rng.choice(questions), language=self.questions_language)
                            if questions is not None
                            else None
                        ),
                    )


@dataclass
class NeMoSFTExample:
    data: dict
    language: str | None = None
    input_ids: np.ndarray | None = None
    context_ids: np.ndarray | None = None
    answer_ids: np.ndarray | None = None
    mask: np.ndarray | None = None
    metadata: dict | None = None

    def tokenize(self, tokenizer: TokenizerWrapper | TokenizerSpec) -> "NeMoSFTExample":
        """
        Create a tokenized variant of this example given a tokenizer (i.e. fill the optional fields).
        Supports BPE tokenizers and aggregate tokenizers.

        The tokenization is compatible with Megatron's :class:`GPTSFTChatDataset`.
        """
        special_tokens = {
            "system_turn_start": "<extra_id_0>",
            "turn_start": "<extra_id_1>",
            "label_start": "<extra_id_2>",
            "end_of_turn": "\n",
            "end_of_name": "\n",
        }

        if isinstance(tokenizer, TokenizerWrapper):
            tokenizer = tokenizer._tokenizer
        if isinstance(tokenizer, AggregateTokenizer):
            assert self.language is not None, (
                f"Error: attempted to use AggregateTokenizer for NeMoSFTExample which did not specify language. "
                f"Problematic example: {self}"
            )
            assert self.language in tokenizer.tokenizers_dict, (
                f"Error: attempted to use AggregateTokenizer for NeMoSFTExample with unsupported language: {self.language}. "
                f"The set of supported languages is: {' '.join(tokenizer.tokenizers_dict.keys())}. "
                f"Problematic example: {self}"
            )
            tokenizer = tokenizer.tokenizers_dict[self.language]

        label_start_tokens, name_end_token_ids, num_turn_start_tokens = _build_samples_mapping(
            tokenizer, special_tokens
        )

        tokenized = preprocess(
            source=self.data,
            tokenizer=tokenizer,
            name_end_token_ids=name_end_token_ids,
            label_start_ids=label_start_tokens,
            special_tokens=special_tokens,
            num_turn_start_tokens=num_turn_start_tokens,
        )
        self.input_ids = tokenized["input_ids"].numpy()
        self.context_ids = tokenized["context_ids"].numpy()
        self.answer_ids = tokenized["answer_ids"].numpy()
        self.mask = tokenized["mask"].numpy()
        self.metadata = {k: v for k, v in self.data.items() if k not in ['conversations']}

        return self

    # TODO(pzelasko): for mini-batch sampling purposes, should we consider input_ids or answer_ids
    #                 as representative of the sequence length? Putting input_ids here for now.

    @property
    def tokens(self) -> np.ndarray:
        return self.input_ids

    @property
    def num_tokens(self) -> int:
        return self.input_ids.shape[0]


@dataclass
class NeMoSFTJsonlAdapter:
    """
    ``NeMoSFTJsonlAdapter`` is used to read a NeMo LM SFT Chat JSONL file and yield objects of type
    ``NeMoSFTExample`` that can be sampled with Lhotse.

    We expect the following schema (contained in a single line per example)::

        {
            "conversations": [
                {
                    "value": str,
                    "from": "User" | "Assistant",
                    "canonical_form": str,
                    "label": str | null
                },
                ...
            ],
            "mask": "User" | "Assistant",
            "system": str,
            "dataset": str,
            "category": str,
        }

    Refer to examples of this format here:

    * TODO: links to examples?
    * TODO: links to more detailed schema definition?
    """

    paths: Union[Pathlike, list[Pathlike]]
    language: str | None = None
    shuffle_shards: bool = False
    shard_seed: Union[int, Literal["trng", "randomized"]] = "trng"

    def __post_init__(self):
        self.paths = expand_sharded_filepaths(self.paths)

    def __iter__(self) -> Iterator[NeMoSFTExample]:
        paths = self.paths
        if self.shuffle_shards:
            seed = resolve_seed(self.shard_seed)
            random.Random(seed).shuffle(paths)
        for path in paths:
            for data in load_jsonl(path):
                yield NeMoSFTExample(data, language=self.language)


@dataclass
class TextTurn:
    value: str
    role: str


@dataclass
class AudioTurn:
    cut: Cut
    role: str
    audio_locator_tag: str


@dataclass
class NeMoMultimodalConversation:
    id: str
    turns: list[TextTurn | AudioTurn]
    input_ids: np.ndarray | None = None
    context_ids: np.ndarray | None = None
    answer_ids: np.ndarray | None = None
    mask: np.ndarray | None = None

    def tokenize(
        self,
        tokenizer: TokenizerWrapper | TokenizerSpec,
        prompt: PromptFormatter = None,
    ) -> "NeMoMultimodalConversation":
        """
        Create a tokenized variant of this example given a tokenizer (i.e. fill the optional fields).
        Supports BPE tokenizers and aggregate tokenizers.

        The tokenization is compatible with Megatron's :class:`GPTSFTChatDataset`.
        """
        if isinstance(tokenizer, TokenizerWrapper):
            tokenizer = tokenizer._tokenizer
        if isinstance(tokenizer, AggregateTokenizer):
            raise NotImplementedError("NeMoMultimodalConversation does not support AggregateTokenizer yet.")
        if prompt is None:
            prompt = PromptFormatter.resolve("plain")(tokenizer)
        elif isinstance(prompt, str):
            prompt = PromptFormatter.resolve(prompt)(tokenizer)

        # Collapse consecutive same-role turns into single turn for proper prompt formatting.
        turns = groupby(
            [
                {
                    "role": turn.role,
                    "slots": {"message": turn.value if isinstance(turn, TextTurn) else turn.audio_locator_tag},
                }
                for turn in self.turns
            ],
            key=lambda turn: turn["role"],
        )
        turns = [
            {"role": role, "slots": {"message": " ".join(t["slots"]["message"] for t in turn_grp)}}
            for role, turn_grp in turns
        ]
        ans = prompt.encode_dialog(turns)
        self.input_ids = ans["input_ids"]
        self.context_ids = ans["context_ids"]
        self.answer_ids = ans["answer_ids"]
        self.mask = ans["mask"]

        return self


@dataclass
class NeMoMultimodalConversationJsonlAdapter:
    """
    ``NeMoMultimodalConversationJsonlAdapter`` is used to read a NeMo multimodal conversation JSONL
    and yield objects of type ``NeMoMultimodalConversation`` that can be sampled with Lhotse.

    We expect the following schema (contained in a single line per example)::

        {
            "id": str,
            "conversations": [
                {
                    "value": str,  # text message or path to audio
                    "from": "User" | "Assistant",
                    "type": "text" | "audio",
                    "duration": float,  # only for audio
                },
                ...
            ],
        }
    """

    manifest_filepath: str | list[str]
    audio_locator_tag: str
    tarred_audio_filepaths: str | list[str] = None
    shuffle_shards: bool = False
    shard_seed: Union[int, Literal["trng", "randomized"]] = "trng"

    def __post_init__(self):
        self.manifest_filepath = expand_sharded_filepaths(self.manifest_filepath)
        if self.tarred_audio_filepaths is not None:
            raise NotImplementedError(
                "Tarred manifests are currently not supported yet for NeMoMultimodalConversation."
            )
            self.tarred_audio_filepaths = expand_sharded_filepaths(self.tarred_audio_filepaths)

    def __iter__(self) -> Iterator[NeMoMultimodalConversation]:
        paths = self.manifest_filepath
        if self.shuffle_shards:
            seed = resolve_seed(self.shard_seed)
            random.Random(seed).shuffle(paths)
        for path in paths:
            for data in load_jsonl(path):
                yield NeMoMultimodalConversation(
                    id=data["id"],
                    turns=[
                        (
                            TextTurn(
                                value=turn["value"],
                                role=turn[
                                    "from"
                                ].lower(),  # prompt formatter role's are typically lowercase: user/assistant
                            )
                            if turn["type"] == "text"
                            else AudioTurn(
                                cut=Recording.from_file(get_full_path(turn["value"], path)).to_cut(),
                                role=turn[
                                    "from"
                                ].lower(),  # prompt formatter role's are typically lowercase: user/assistant
                                audio_locator_tag=self.audio_locator_tag,
                            )
                        )
                        for turn in data["conversations"]
                    ],
                )


"""
The code below is copied from nemo/collections/nlp/data/language_modeling/megatron/gpt_sft_chat_dataset.py
with minimal modifications in order to avoid importing the NLP collection.

We require this code for on-the-fly text example tokenization in a compatible way with Megatron,
so that we can determine the mini-batch sizes using the token counts.
"""


def preprocess(
    source: dict,
    tokenizer: TokenizerSpec,
    name_end_token_ids: int,
    label_start_ids: list,
    special_tokens: dict,
    num_turn_start_tokens: int,
):
    """
    Given a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    header, conversation, data_type, mask_role = _get_header_conversation_type_mask_role(source, special_tokens)
    # tokenize conversations
    input_ids = tokenizer.text_to_ids(conversation)
    target = copy.deepcopy(input_ids)
    header_tokens = tokenizer.text_to_ids(header)
    header_len = len(header_tokens)

    ids = []
    tokenized_lens = []
    assert torch.equal(torch.tensor(target[:header_len]), torch.tensor(header_tokens))
    for s in source['conversations']:
        # hack to remove the extra empty token in front
        id1 = tokenizer.text_to_ids(PREFIX_STR + s["value"])
        id2 = tokenizer.text_to_ids(PREFIX_STR)
        tokenized_sentence = id1[len(id2) :]
        ids.append(torch.tensor(tokenized_sentence))
        tokenized_lens.append(len(tokenized_sentence))
    speakers = [sentence["from"] for sentence in source['conversations']]
    assert mask_role in speakers, "mask role not in the conversation"
    target = torch.LongTensor(target)
    # not going to train on the header
    target[:header_len] = IGNORE_INDEX
    input_ids = torch.LongTensor(input_ids)
    _mask_targets(
        target,
        tokenized_lens,
        speakers,
        header_len,
        ids,
        tokenizer,
        mask_role,
        data_type,
        name_end_token_ids,
        special_tokens,
        label_start_ids,
        num_turn_start_tokens,
    )
    mask = (target != IGNORE_INDEX).bool()
    assert mask.sum().item() != 0, "mask is empty"
    # Choose the last conversation as answer other history are context
    last_ignore_index_pos = torch.nonzero(target == IGNORE_INDEX)[-1].item() + 1
    context_ids = input_ids[:last_ignore_index_pos]
    answer_ids = input_ids[last_ignore_index_pos:]
    return dict(input_ids=input_ids, mask=mask, context_ids=context_ids, answer_ids=answer_ids)


def _build_samples_mapping(tokenizer, special_tokens):
    # Copied from gpt_sft_chat_dataset.py
    LABEL_START = special_tokens['label_start']
    END_NAME_SIGNAL = special_tokens['end_of_name']

    id1 = tokenizer.text_to_ids(PREFIX_STR)
    id2 = tokenizer.text_to_ids(PREFIX_STR + LABEL_START)
    label_start_tokens = id2[len(id1) :]

    id1 = tokenizer.text_to_ids(PREFIX_STR + END_NAME_SIGNAL)
    id2 = tokenizer.text_to_ids(PREFIX_STR)
    name_end_token_ids = id1[len(id2) :]

    id1 = tokenizer.text_to_ids(PREFIX_STR + special_tokens['turn_start'])
    id2 = tokenizer.text_to_ids(PREFIX_STR)
    num_turn_start_tokens = len(id1) - len(id2)

    return label_start_tokens, name_end_token_ids, num_turn_start_tokens


PREFIX_STR = (
    "\x00"  # the prefix string used in the tokenizer to deal with the added empty token for some of the tokenizers
)

IGNORE_INDEX = -100
SYSTEM_TOKEN = "System"

TYPE_INSTRUCTION = {
    'TEXT_TO_VALUE': "",
    'VALUE_TO_TEXT': '',
}


def _get_header_conversation_type_mask_role(source, special_tokens):
    END_SIGNAL = special_tokens['end_of_turn']
    END_NAME_SIGNAL = special_tokens['end_of_name']

    data_type = None
    if 'type' in source:
        data_type = source['type']
        if data_type is not None:
            assert data_type in TYPE_INSTRUCTION, f"source type {data_type} not supported"
    # add end signal and concatenate together
    conversation = source['system']
    if data_type is not None:
        if TYPE_INSTRUCTION[data_type] != '':
            conversation = conversation + '\n' + TYPE_INSTRUCTION[data_type]
    mask_role = source.get('mask', 'User')
    header = f"{special_tokens['system_turn_start']}{SYSTEM_TOKEN}{END_NAME_SIGNAL}{conversation}{END_SIGNAL}"
    conversation = _add_speaker_and_signal(header, source['conversations'], mask_role, data_type, special_tokens)
    return header, conversation, data_type, mask_role


def identify_start_index_of_subsequence(subsequence, sequence):
    """find the location of the small tensor in the large tensor.
        e.g.  small = [1,3], large = [2,3,1,3], returns 2
              small = [3,2], large = [2,3,1,3], returns -1
    Args:
        small (tensor): small tensor
        large (tensor): large tensor
    """
    for i in range(sequence.size(0) - subsequence.size(0) + 1):
        if torch.equal(sequence[i : i + subsequence.size(0)], subsequence):
            return i
    return -1


def _mask_targets(
    target,
    tokenized_lens,
    speakers,
    header_len,
    s_ids,
    tokenizer,
    mask_role,
    gtype,
    name_end_token_ids,
    special_tokens,
    label_start_ids,
    num_turn_start_tokens,
):
    """This function masks the tokens so the loss is computed only on the non-masked role's responses.
    For 'TEXT_TO_VALUE' type, the loss is computed on the value attributes.

    Args:
        target (Tensor): input ids
        tokenized_lens (List[int]): array of lengths of each turns
        speakers (List[str]): array of speakers of each turns
        header_len (int): the system prompt length
        s_ids (List[Tensor]): array of tokenized ids of each turns
        tokenizer (TokenizerSpec): tokenizer object
        mask_role (str): the speaker id to be masked from loss computation
        gtype (str): either 'TEXT_TO_VALUE' or 'VALUE_TO_TEXT'
        name_end_token_ids (int): end of name token ids
        special_tokens (dict): special tokens used for the chat prompt. It has the keys: system_turn_start, turn_start, label_start, end_of_turn
        label_start_ids (list): list of label start token ids,
        num_turn_start_tokens (int): number of tokens of the turn_start str
    """
    TURN_TOKEN = special_tokens['turn_start']
    END_NAME_SIGNAL = special_tokens['end_of_name']
    label_start_ids = torch.tensor(label_start_ids)
    name_end_token_ids = torch.tensor(name_end_token_ids)

    cur_idx = header_len
    tgt_len = target.shape[0]
    for i, (tokenized_len, speaker, s_id) in enumerate(zip(tokenized_lens, speakers, s_ids)):
        # note, sentence piece will add extra empty token in front. has to compute the diff
        id1 = tokenizer.text_to_ids(PREFIX_STR)
        id2 = tokenizer.text_to_ids(PREFIX_STR + TURN_TOKEN + speaker + END_NAME_SIGNAL)
        skip_name_len = len(id2) - len(
            id1
        )  # s_ids[:skip_name_len] is the name part of the prompt 'TURN_TOKEN + speaker + END_NAME_SIGNAL'
        # get the position of the label start string in this turn
        location = identify_start_index_of_subsequence(label_start_ids, s_id)

        if location >= 0:
            # if it contains the label start tokens
            if gtype == 'VALUE_TO_TEXT':
                # handles the case that condition on labels to generate respone
                # the next token after the name part of the prompt is the beginning of the label start tokens
                assert skip_name_len == location
                # find the first new line token after the label part, which indicates the end of the whole label string
                # newline_loc = torch.where((s_id[skip_name_len:] == name_end_token_ids))[0]
                newline_loc = identify_start_index_of_subsequence(name_end_token_ids, s_id[skip_name_len:])
                if newline_loc < 0:
                    # cannot find new line token, which means the the whole turn is just a partial label string. Mask the whole turn
                    target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
                    continue
                # skip the label part and the new line token
                more_skip_len = newline_loc + len(name_end_token_ids)
                # skip the name part and the label part
                skip_name_len += more_skip_len
            elif gtype == 'TEXT_TO_VALUE':
                # handles the case that condition on response to generate label
                # skip the name part, response and the label start tokens part, the remainder is the label string without label start, e.g. 'quality:9,toxicity:8...'
                skip_name_len = location + len(label_start_ids)
        if cur_idx >= tgt_len:
            break
        # elif cur_idx + tokenized_len < tgt_len:
        #     # Check whether the mask is applied to the correct position, the first token is turn start tokens
        #     if not torch.equal(target[cur_idx + 1 : cur_idx + tokenized_len], s_id[1:]):
        #         logging.warning("a sentence mismatches the corresponding piece " "in the conversation")
        if i == 0 and (gtype == 'VALUE_TO_TEXT' or gtype is None):
            # mask the first turn completely to provide at least one turn as context for the rest
            target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker == mask_role and i == 1 and gtype == 'TEXT_TO_VALUE':
            # leave the first turn start tag unmasked, servers severs as the end of turn signal
            target[cur_idx + num_turn_start_tokens : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker == mask_role and (i > 1):
            # leave the first turn start tag unmasked, which severs as the end of turn signal
            target[cur_idx + num_turn_start_tokens : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker == mask_role and (i <= 1):
            # mask out everything in the second turn
            target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
        else:
            # mask up to name part, label part for VALUE_TO_TEXT, or name part, response and label start tokens for TEXT_TO_VALUE, or just the name part if gtype is None
            target[cur_idx : cur_idx + skip_name_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, mask_role, gtype, special_tokens):
    TURN_TOKEN = special_tokens['turn_start']
    END_SIGNAL = special_tokens['end_of_turn']
    LABEL_START = special_tokens['label_start']
    END_NAME_SIGNAL = special_tokens['end_of_name']

    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = ""
    conversation = header
    for i, sentence in enumerate(source):
        sentence_from = sentence["from"]
        role_token = TURN_TOKEN
        if gtype is None:
            sentence["value"] = (
                BEGIN_SIGNAL + role_token + sentence_from + END_NAME_SIGNAL + sentence["value"] + END_SIGNAL
            )
        elif gtype == "VALUE_TO_TEXT":
            sentence["value"] = (
                BEGIN_SIGNAL
                + role_token
                + sentence_from
                + END_NAME_SIGNAL
                + (
                    response_value_formater(sentence['label'], LABEL_START, END_NAME_SIGNAL)
                    if 'label' in sentence
                    else ''
                )
                + sentence["value"]
                + END_SIGNAL
            )
        elif gtype == "TEXT_TO_VALUE":
            sentence["value"] = (
                BEGIN_SIGNAL
                + role_token
                + sentence_from
                + END_NAME_SIGNAL
                + sentence["value"]
                + END_SIGNAL
                + (
                    response_value_formater(sentence['label'], LABEL_START, END_NAME_SIGNAL)
                    if 'label' in sentence
                    else ''
                )
            )
        else:
            raise ValueError(
                f"source type {gtype} not supported, only 'VALUE_TO_TEXT' and 'TEXT_TO_VALUE' are supported"
            )
        conversation += sentence["value"]
        # if the last turn is not masked, add next token start token to the end, which will be included for loss calculation
        if sentence_from != mask_role and i == len(source) - 1:
            conversation += TURN_TOKEN
    return conversation


def response_value_formater(label, label_start, end_signal):
    if isinstance(label, str):
        return label_start + label + end_signal
    elif label is None:
        return ''
    else:
        raise ValueError(f'Unknown label type {type(label)}, only str type is supported')
