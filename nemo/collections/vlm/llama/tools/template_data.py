# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from ..prompt_templates import (
    BuiltinToolGenerator,
    JsonCustomToolGenerator,
    ToolResponseGenerator,
)
from .datatypes import BuiltinTool, StopReason, ToolCall

INSTRUCTION = "You are a helpful assistant."


def system_message_builtin_tools_only():
    return {
        "builtin_tools": BuiltinToolGenerator().data_examples()[0],
        "custom_tools": [],
        "instruction": INSTRUCTION,
    }


def system_message_custom_tools_only():
    return {
        "builtin_tools": [],
        "custom_tools": JsonCustomToolGenerator().data_examples()[0],
        "instruction": INSTRUCTION,
    }


def system_message_builtin_and_custom_tools():
    return {
        "builtin_tools": BuiltinToolGenerator().data_examples()[0],
        "custom_tools": JsonCustomToolGenerator().data_examples()[0],
        "instruction": INSTRUCTION,
    }


def system_default():
    return {
        "builtin_tools": [],
        "custom_tools": [],
        "instruction": INSTRUCTION,
    }


def tool_success():
    return ToolResponseGenerator().data_examples()[0]


def tool_failure():
    return ToolResponseGenerator().data_examples()[1]


def assistant_builtin_tool_call():
    return {
        "content": "",
        "tool_call": ToolCall(
            call_id="uuid",
            tool_name=BuiltinTool.brave_search,
            arguments={
                "query": "Who won NBA in 2024?",
            },
        ),
        "stop_reason": StopReason.end_of_message,
    }


def assistant_custom_tool_call():
    return {
        "content": "",
        "tool_call": ToolCall(
            call_id="uuid",
            tool_name="trending_songs",
            arguments={"country": "US", "n": 10},
        ),
        "stop_reason": StopReason.end_of_turn,
    }


def assistant_default():
    return {
        "content": "Hi, I am a helpful assistant. What can I help you with today?",
        "tool_call": None,
        "stop_reason": StopReason.end_of_turn,
    }


def user_default():
    return {"content": "Please tell me how to plan a trip to New York"}


def user_images():
    return {"content": "<|image|><|image|>What do these images depict?"}


def user_interleaved_images():
    return {
        "content": "<|image|>Describe the image in one sentence.<|image|>Write a haiku about these images"
    }
