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

# NOTE: This script is just an example of using NeMo checkpoints
# for generating outputs and is subject to change without notice.

PREFIX_STR = (
    "\x00"  # the prefix string used in the tokenizer to deal with the added empty token for some of the tokenizers
)

IGNORE_INDEX = -100
SYSTEM_TOKEN = "System"

TYPE_INSTRUCTION = {
    'TEXT_TO_VALUE': "",
    'VALUE_TO_TEXT': '',
}

NM5_CHAT_PROMPT_TOKENS={
  "system_turn_start": "<SPECIAL_10>",
  "turn_start": "<SPECIAL_11>",
  "label_start": "<SPECIAL_12>",
  "end_of_turn": "\n",
  "end_of_name": "\n"
}
def response_value_formater(label, label_start, end_signal):
    if isinstance(label, str):
        return label_start + label + end_signal
    elif label is None:
        return ''
    else:
        raise ValueError(f'Unknown label type {type(label)}, only str type is supported')

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
        # if the last turn is not masked, add next token start token to the end,
        # which will be included for loss calculation
        if sentence_from not in mask_role and i == len(source) - 1:
            conversation += TURN_TOKEN
    return conversation

def get_header_conversation_type_mask_role(source, special_tokens):
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

def convert_messages(input_list):
        output_dict = {
            'system': '',
            'conversations': [],
            'mask': 'User',
            'type': 'VALUE_TO_TEXT',
        }

        # Extract the system message
        for msg in input_list:
            if msg['role'] == 'system':
                output_dict['system'] = msg['content']
                break  # Assuming only one system message

        # Build the conversations list
        for msg in input_list:
            if msg['role'] != 'system':
                conversation_entry = {
                    'from': msg['role'].capitalize(),  # Capitalize 'user' and 'assistant'
                    'value': msg['content'],
                    'label': None,
                }
                output_dict['conversations'].append(conversation_entry)

        return output_dict
