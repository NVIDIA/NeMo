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

import copy
import json

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm.gpt.data.chat import ChatDataModule
from nemo.collections.llm.gpt.data.core import GPTSFTChatDataset, create_sft_dataset
from nemo.collections.llm.gpt.data.utils import _chat_preprocess

LLAMA_31_CHAT_TEMPLATE_WITH_GENERATION_TAGS = """{{- bos_token }}
{%- if not date_string is defined %}
    {%- set date_string = "30 Aug 2024" %}
{%- endif %}
{%- set loop_messages = messages %}
{%- if tools is not none and tool_choice is not none %}
    {{- '<|start_header_id|>system<|end_header_id|>\n\n' }}
    {{- "Environment: ipython\n\n" }}
    {{- "Cutting Knowledge Date: December 2023\n" }}
    {{- "Today Date: " + date_string + "\n\n" }}
    {{- "You are a helpful assistant.\n" }}
    {{- '<|eot_id|>' }}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' }}
    {{- 'You have access to the following functions to supplement your existing knowledge:\n\n' }}
    {%- for t in tools %}
        {%- set tname = t.function.name %}
        {%- set tdesc = t.function.description %}
        {%- set tparams = t.function.parameters | tojson %}
        {{- "Use the function '" + tname + "' to '" + tdesc + "':\n" }}
        {{- '{"name": "' + tname + '", "description": "' + tdesc + '", "parameters": ' + tparams + '}\n\n' }}
    {%- endfor %}
    {{- 'Think very carefully before calling functions.\n' }}
    {{- 'Only call them if they are relevant to the prompt.\n' }}
    {{- 'If you choose to call a function ONLY reply in the following format with no natural ' }}
    {{- 'language surrounding it:\n\n' }}
    {{- '<function=example_function_name>{"example_name": "example_value"}</function>\n\n' }}
    {{- 'Reminder:\n' }}
    {{- '- Function calls MUST follow the specified format, start with <function= and end with </function>\n' }}
    {{- '- Required parameters MUST be specified\n' }}
    {{- '- Only call one function at a time\n' }}
    {{- '- Put the entire function call reply on one line\n' }}
    {{- '- Do not call functions if they are not relevant to the prompt' }}
    {{- '<|eot_id|>' }}
{%- endif %}
{%- for message in loop_messages %}
    {%- if message['role'] in ['ipython', 'tool'] %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {{- "[stdout]" + message['content'] | trim  + "[/stdout]\n<|eot_id|>" }}
    {%- elif message['role'] == 'assistant'%}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
        {%- if message.get('tool_calls') is not none %}
            {%- set tool_call = message['tool_calls'][0] %}
            {%- generation %}
                {{- '<|python_tag|><function=' + tool_call.function.name + '>' }}
                {{- tool_call.function.arguments | tojson + '</function>\n<|eot_id|>' }}
            {%- endgeneration %}
        {%- else %}
            {%- generation %}
                {{- message['content'] | trim + '<|eot_id|>' }}
            {%- endgeneration %}
        {%- endif %}
    {%- else %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
        {{- message['content'] | trim + '<|eot_id|>' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""


@pytest.fixture
def mock_tokenizer():
    tokenizer = AutoTokenizer(
        pretrained_model_name="/home/TestData/nemo2_ckpt/Llama3Config8B",
        use_fast=True,
        chat_template=LLAMA_31_CHAT_TEMPLATE_WITH_GENERATION_TAGS,
    )
    return tokenizer


@pytest.fixture
def mock_trainer():
    trainer = MagicMock()
    trainer.global_step = 0
    trainer.max_steps = 1000
    return trainer


@pytest.fixture
def sample_chat_dataset():
    return Dataset.from_dict(
        {
            "conversations": [
                [
                    {"from": "human", "value": "Hello, how are you?"},
                    {"from": "assistant", "value": "I'm doing well, thank you! How can I help you today?"},
                ],
                [
                    {"from": "human", "value": "What's the weather like?"},
                    {"from": "assistant", "value": "I don't have access to real-time weather information."},
                ],
            ]
        }
    )


@pytest.fixture
def temp_dataset_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def chat_data_module(mock_tokenizer, temp_dataset_dir):
    with patch('nemo.collections.llm.gpt.data.core.create_sft_dataset') as mock_create_dataset:
        mock_create_dataset.return_value = MagicMock()
        data_module = ChatDataModule(
            tokenizer=mock_tokenizer,
            seq_length=512,
            micro_batch_size=2,
            global_batch_size=4,
            dataset_root=temp_dataset_dir,
        )
        return data_module


def test_chat_data_module_initialization(chat_data_module):
    assert chat_data_module.seq_length == 512
    assert chat_data_module.micro_batch_size == 2
    assert chat_data_module.global_batch_size == 4


def test_create_dataset(chat_data_module, temp_dataset_dir):
    # Create a sample chat dataset file
    dataset_path = temp_dataset_dir / "chat_dataset.jsonl"
    with open(dataset_path, "w") as f:
        json.dump(
            {"conversations": [[{"from": "human", "value": "Hello"}, {"from": "assistant", "value": "Hi there!"}]]}, f
        )

    # Test dataset creation
    dataset = chat_data_module._create_dataset(str(dataset_path))
    assert dataset is not None

    # Test with is_test=True
    test_dataset = chat_data_module._create_dataset(str(dataset_path), is_test=True)
    assert test_dataset is not None


def test_create_dataset_with_hf_template(temp_dataset_dir, mock_tokenizer):

    dataset_path = temp_dataset_dir / "chat_dataset.jsonl"
    with open(dataset_path, "w") as f:
        json.dump(
            {
                "messages": [
                    {"role": "system", "content": "you are a robot"},
                    {"role": "user", "content": "Choose a number that is greater than 0 and less than 2\n"},
                    {
                        "role": "assistant",
                        "content": "2",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": {"location": "Denver"}},
                            },
                            # additional tool calls should be quietly ignored
                            {
                                "type": "function",
                                "function": {"name": "extra", "arguments": {"non-existing": "tool call"}},
                            },
                        ],
                    },
                ]
            },
            f,
        )

    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Fetches the current weather for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "The name of the location."}},
                    "required": ["location"],
                },
            },
        }
    ]

    dataset = create_sft_dataset(
        path=dataset_path,
        tokenizer=mock_tokenizer,
        seq_length=512,
        prompt_template="{input} {output}",
        chat=True,
        use_hf_tokenizer_chat_template=True,
        tool_schemas=tool_schemas,
    )

    assert isinstance(dataset, GPTSFTChatDataset)
    assert dataset.max_seq_length == 512

    assert dataset.tool_schemas == tool_schemas

    data = [dataset[idx] for idx in list(range(0, len(dataset)))]

    collated = dataset.collate_fn(data)
    assert len(collated["tokens"]) == len(data)

    tokens = collated["tokens"][0]
    loss = collated["loss_mask"][0]

    tokens_copy = copy.deepcopy(tokens)
    tokens_copy[loss == 0] = mock_tokenizer.text_to_ids("-")[0]

    hf_tokenizer = mock_tokenizer.tokenizer
    full_string = hf_tokenizer.convert_tokens_to_string(mock_tokenizer.ids_to_tokens(tokens))
    assistant_message_only = hf_tokenizer.convert_tokens_to_string(mock_tokenizer.ids_to_tokens(tokens_copy))

    assert full_string == (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nEnvironment: ipython\n\n"
        "Cutting Knowledge Date: December 2023\nToday Date: 30 Aug 2024\n\n"
        "You are a helpful assistant.\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "You have access to the following functions to supplement your existing knowledge:\n\n"
        "Use the function 'get_weather' to 'Fetches the current weather for a given location.':\n"
        '{"name": "get_weather", "description": "Fetches the current weather for a given location.", '
        '"parameters": {"type": "object", "properties": {"location": {"type": "string", '
        '"description": "The name of the location."}}, "required": ["location"]}}\n\n'
        "Think very carefully before calling functions.\n"
        "Only call them if they are relevant to the prompt.\n"
        "If you choose to call a function ONLY reply in the following format with no natural language "
        "surrounding it:\n\n"
        "<function=example_function_name>{\"example_name\": \"example_value\"}</function>\n\n"
        "Reminder:\n"
        "- Function calls MUST follow the specified format, start with <function= and end with </function>\n"
        "- Required parameters MUST be specified\n"
        "- Only call one function at a time\n"
        "- Put the entire function call reply on one line\n"
        "- Do not call functions if they are not relevant to the prompt"
        "<|eot_id|><|start_header_id|>system<|end_header_id|>\n\nyou are a robot"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "Choose a number that is greater than 0 and less than 2"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "<|python_tag|><function=get_weather>{\"location\": \"Denver\"}</function>\n"
        "<|eot_id|><|end_of_text|>"
    )

    assert assistant_message_only == (
        "------------------------------------------------------------------------------------------------------------"
        "------------------------------------------------------------------------------------------------------------"
        "-------------------------------------------------------------------\n\n"
        """<|python_tag|><function=get_weather>{"location": "Denver"}</function>\n"""
        "<|eot_id|>-"
    )


class TestPreprocess:
    tokenizer = AutoTokenizer(
        pretrained_model_name="/home/TestData/nemo2_ckpt/Llama3Config8B",
        use_fast=True,
        chat_template=LLAMA_31_CHAT_TEMPLATE_WITH_GENERATION_TAGS,
    )
    conversations = {
        "system": "you are a robot",
        "conversations": [
            {"from": "User", "value": "Choose a number that is greater than 0 and less than 2\n"},
            {"from": "Assistant", "value": "1"},
        ],
    }
    messages = {
        "messages": [
            {"role": "system", "content": "you are a robot"},
            {"role": "user", "content": "Choose a number that is greater than 0 and less than 2\n"},
            {"role": "assistant", "content": "1"},
        ]
    }
    # fmt: off
    # Example tokenized output data from `_chat_preprocess` for the above messages
    output_data = {
        # input_ids contain tokenized messages with chat template applied
        # Note: the NIM chat template did not include the date_time or knowledge cutoff date_string
        # for non-tool calling applications
        'input_ids': [
            128000, 128006, 9125, 128007, 271, 9514, 527, 264, 12585, 128009, 128006, 882, 128007, 271,
            25017, 264, 1396, 430, 374, 7191, 1109, 220, 15, 323, 2753, 1109, 220, 17, 128009, 128006, 78191,
            128007, 271, 16, 128009, 128001
        ],
        # mask corresponds to tokens of input_ids where 1 represents output tokens for the role `assistant` in both
        # context and answer for multi-turn, and 0 to mask all other tokens, e.g. system, user, and tool calling.
        'loss_mask': [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1
        ],
        # context_ids contain tokenized messages with chat template applied for all messages except assistant's last
        # generated output
        'context_ids': [
            128000, 128006, 9125, 128007, 271, 9514, 527, 264, 12585, 128009, 128006, 882, 128007, 271, 25017, 264,
            1396, 430, 374, 7191, 1109, 220, 15, 323, 2753, 1109, 220, 17, 128009, 128006, 78191, 128007, 271
        ],
        # answer_ids contain tokenized messages with chat template applied for only the assistant's last generated
        # output
        'answer_ids': [16, 128009, 128001]
    }
    # fmt: on
    decoded_context = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

you are a robot<|eot_id|><|start_header_id|>user<|end_header_id|>

Choose a number that is greater than 0 and less than 2<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    decoded_answer = "1<|eot_id|><|end_of_text|>"

    def test_nemo_format(self):
        tokenized_chat = _chat_preprocess(self.conversations, self.tokenizer)
        decoded_input = self.tokenizer.tokenizer.decode(tokenized_chat["input_ids"])
        decoded_context = self.tokenizer.tokenizer.decode(tokenized_chat["context_ids"])
        decoded_answer = self.tokenizer.tokenizer.decode(tokenized_chat["answer_ids"])
        assert torch.equal(
            tokenized_chat["input_ids"],
            torch.cat((tokenized_chat["context_ids"], tokenized_chat["answer_ids"]), dim=-1),
        )
        assert self.decoded_context == decoded_context
        assert self.decoded_answer == decoded_answer
        assert self.decoded_context + self.decoded_answer == decoded_input
        assert torch.equal(
            torch.LongTensor(self.output_data["input_ids"]),
            tokenized_chat["input_ids"],
        )
        assert torch.equal(
            torch.LongTensor(self.output_data["context_ids"]),
            tokenized_chat["context_ids"],
        )
        assert torch.equal(
            torch.LongTensor(self.output_data["context_ids"]),
            tokenized_chat["context_ids"],
        )
        assert torch.equal(
            torch.LongTensor(self.output_data["loss_mask"]),
            tokenized_chat["loss_mask"],
        )

    def test_messages_format(self):
        tokenized_chat = _chat_preprocess(self.messages, self.tokenizer)
        decoded_input = self.tokenizer.tokenizer.decode(tokenized_chat["input_ids"])
        decoded_context = self.tokenizer.tokenizer.decode(tokenized_chat["context_ids"])
        decoded_answer = self.tokenizer.tokenizer.decode(tokenized_chat["answer_ids"])

        assert torch.equal(
            tokenized_chat["input_ids"],
            torch.cat((tokenized_chat["context_ids"], tokenized_chat["answer_ids"]), dim=-1),
        )
        assert self.decoded_context == decoded_context
        assert self.decoded_answer == decoded_answer
        assert self.decoded_context + self.decoded_answer == decoded_input
        assert torch.equal(
            torch.LongTensor(self.output_data["input_ids"]),
            tokenized_chat["input_ids"],
        )
        assert torch.equal(
            torch.LongTensor(self.output_data["context_ids"]),
            tokenized_chat["context_ids"],
        )
        assert torch.equal(
            torch.LongTensor(self.output_data["context_ids"]),
            tokenized_chat["context_ids"],
        )
        assert torch.equal(
            torch.LongTensor(self.output_data["loss_mask"]),
            tokenized_chat["loss_mask"],
        )

    def test_mask_no_assistant(self):
        messages = copy.deepcopy(self.messages)
        messages["messages"][2]["role"] = "not-assistant"
        tokenized_chat = _chat_preprocess(messages, self.tokenizer)
        assert sum(tokenized_chat["loss_mask"]) == 1, "No matching 'assistant' role to mask"
        assert len(tokenized_chat["answer_ids"]) == 1

    def test_multi_turn(self):
        messages = {
            "messages": [
                {"role": "user", "content": "Choose a number that is greater than 0 and less than 2\n"},
                {"role": "assistant", "content": "1"},
                {"role": "user", "content": "Choose another number that is greater than 0 and less than 3\n"},
                {"role": "assistant", "content": "2"},
            ]
        }
        tokenized_chat = _chat_preprocess(messages, self.tokenizer)
        decoded_context = self.tokenizer.tokenizer.decode(tokenized_chat["context_ids"])
        decoded_answer = self.tokenizer.tokenizer.decode(tokenized_chat["answer_ids"])
        # Verify all assistant outputs appear in mask
        assistant_mask = [i for i, mask in enumerate(tokenized_chat["loss_mask"]) if mask == 1]
        assistant_generated_tokens = [tokenized_chat["input_ids"][idx] for idx in assistant_mask]
        assert "1<|eot_id|>2<|eot_id|><|end_of_text|>" == self.tokenizer.tokenizer.decode(assistant_generated_tokens)
        # Verify context includes assistant output
        assert decoded_context.count("<|start_header_id|>assistant<|end_header_id|>") == 2
        # Verify answer only includes last assistant output and does not contain assistant header
        assert "<|start_header_id|>assistant<|end_header_id|>" not in decoded_answer
        assert "2<|eot_id|><|end_of_text|>" == decoded_answer

    def test_tool_calling(self):
        messages = {
            "messages": [
                {"role": "user", "content": "Choose a number that is greater than 0 and less than 2\n"},
                {"role": "assistant", "content": "1"},
                {"role": "user", "content": "What's the weather in Denver today?\n"},
                {
                    "role": "assistant",
                    "content": "2",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "get_weather", "arguments": {"location": "Denver"}}},
                        # additional tool calls should be quietly ignored
                        {
                            "type": "function",
                            "function": {"name": "extra", "arguments": {"non-existing": "tool call"}},
                        },
                    ],
                },
                {"role": "tool", "content": '{"Denver": {"temperature": "72°F"}}'},
                {"role": "assistant", "content": "The current weather in Denver is 72°F and sunny."},
            ]
        }
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Fetches the current weather for a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string", "description": "The name of the location."}},
                        "required": ["location"],
                    },
                },
            }
        ]

        tokenized_chat = _chat_preprocess(messages, self.tokenizer, tool_schemas)
        decoded_context = self.tokenizer.tokenizer.decode(tokenized_chat["context_ids"])
        decoded_answer = self.tokenizer.tokenizer.decode(tokenized_chat["answer_ids"])
        # Verify all assistant outputs appear in mask
        assistant_mask = [i for i, mask in enumerate(tokenized_chat["loss_mask"]) if mask == 1]
        assistant_generated_tokens = [tokenized_chat["input_ids"][idx] for idx in assistant_mask]
        assert (
            '1<|eot_id|><|python_tag|><function=get_weather>{"location": "Denver"}</function>\n<|eot_id|>'
            'The current weather in Denver is 72°F and sunny.<|eot_id|><|end_of_text|>'
            == self.tokenizer.tokenizer.decode(assistant_generated_tokens)
        )
        # Verify context includes assistant output
        assert decoded_context.count("<|start_header_id|>assistant<|end_header_id|>") == 3
        assert decoded_context.count("<|start_header_id|>system<|end_header_id|>") == 1
        assert (
            """Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- Do not call functions if they are not relevant to the prompt<|eot_id|>"""
            in decoded_context
        )
        assert (
            "Use the function 'get_weather' to 'Fetches the current weather for a given location.'" in decoded_context
        )
        # Verify answer only includes last assistant output and does not contain assistant header
        assert "<|start_header_id|>assistant<|end_header_id|>" not in decoded_answer
        assert "The current weather in Denver is 72°F and sunny.<|eot_id|><|end_of_text|>" == decoded_answer

        # Verify that Specifying a chat's tools schema overrides the default tool_schema
        messages["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "launch",
                    "description": "Trigger the eotw sequence",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string", "description": "The name of the location."}},
                        "required": ["location"],
                    },
                },
            }
        ]
        messages["messages"][3]["tool_calls"][0]["function"]["name"] = "launch"
        messages["messages"][-1]["content"] = "Starting the eotw at Denver! Have a nice Day."
        tokenized_chat = _chat_preprocess(messages, self.tokenizer, tool_schemas)
        decoded_context = self.tokenizer.tokenizer.decode(tokenized_chat["context_ids"])
        decoded_answer = self.tokenizer.tokenizer.decode(tokenized_chat["answer_ids"])
        assistant_mask = [i for i, mask in enumerate(tokenized_chat["loss_mask"]) if mask == 1]
        assistant_generated_tokens = [tokenized_chat["input_ids"][idx] for idx in assistant_mask]
        assert (
            '1<|eot_id|><|python_tag|><function=launch>{"location": "Denver"}</function>\n<|eot_id|>'
            'Starting the eotw at Denver! Have a nice Day.<|eot_id|><|end_of_text|>'
            == self.tokenizer.tokenizer.decode(assistant_generated_tokens)
        )
        assert "<|start_header_id|>assistant<|end_header_id|>" not in decoded_answer
        assert "Starting the eotw at Denver! Have a nice Day.<|eot_id|><|end_of_text|>" == decoded_answer
        assert (
            "Use the function 'get_weather' to 'Fetches the current weather for a given location.'"
            not in decoded_context
        )
        assert "Use the function 'launch' to 'Trigger the eotw sequence'" in decoded_context
