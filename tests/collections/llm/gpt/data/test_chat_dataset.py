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

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from datasets import Dataset

from nemo.collections.llm.gpt.data.chat import ChatDataModule


def mock_llama_apply_chat_template(
    messages,
    tools=None,
    tokenize=False,
    return_dict=False,
    return_assistant_tokens_mask=False,
    add_generation_prompt=False,
):
    """
    Mock version of transformers.ChatTemplate.apply_chat_template for LLaMA-style chat.

    Parameters:
        messages (list): List of dicts with 'role' and 'content'.
        tokenize (bool): Ignored here — included for API compatibility.
        add_generation_prompt (bool): If True, leaves the last assistant message blank for generation.
        return_dict (bool):
            If True, return a dictionary with keys like 'input_ids', 'labels', etc.
            If False, return a list of values in the order: [input_ids, attention_mask, labels, (assistant_masks)].
        return_assistant_tokens_mask (bool):
            If True, include 'assistant_masks' in the output. This is a list where positions corresponding
            to assistant response tokens are marked as 1 (used to identify loss-relevant tokens).
    Returns:
        str || Dict: Chat string formatted for LLaMA models or dictionary of response
    """
    # === Step 1: Build prompt string ===
    prompt = ""
    segments = []  # Track which tokens belong to which role
    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            prompt += "<|system|>\n" + content + "\n"
            segments.append(("system", content))
        elif role == "user":
            prompt += "<|user|>\n" + content + "\n"
            segments.append(("user", content))
        elif role == "assistant":
            is_last = i == len(messages) - 1
            if is_last and add_generation_prompt:
                prompt += "<|assistant|>\n"
                segments.append(("assistant", ""))
            else:
                prompt += "<|assistant|>\n" + content + "\n"
                segments.append(("assistant", content))

    if not tokenize:
        return prompt

    # === Step 2: Fake tokenization ===
    tokens = []
    roles = []
    for role, content in segments:
        for word in content.strip().split():
            tokens.append(f"<{word}>")  # simulate a token
            roles.append(role)

    input_ids = [100 + i for i in range(len(tokens))]
    attention_mask = [1] * len(input_ids)

    # === Step 3: Compute loss mask via labels
    labels = []
    assistant_token_mask = []
    for role in roles:
        if role == "assistant":
            labels.append(1)  # placeholder value, will map to input_ids later
            assistant_token_mask.append(1)
        else:
            labels.append(-100)
            assistant_token_mask.append(0)

    # Replace placeholder `1` in labels with correct token IDs
    labels = [token_id if mask == 1 else -100 for token_id, mask in zip(input_ids, assistant_token_mask)]

    # === Step 4: Build return dict ===
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    if return_assistant_tokens_mask:
        result["assistant_masks"] = assistant_token_mask

    return result if return_dict else list(result.values())


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.text_to_ids.side_effect = lambda text: [ord(c) for c in text]  # Mock character-based token IDs
    tokenizer.bos_id = 1
    tokenizer.eos_id = 2
    tokenizer.tokenizer = MagicMock()
    tokenizer.tokenizer.apply_chat_template = mock_llama_apply_chat_template

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
