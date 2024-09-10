# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import json
import re
from typing import Optional, Tuple

from .datatypes import BuiltinTool, ToolCall, ToolPromptFormat

BUILTIN_TOOL_PATTERN = r'\b(?P<tool_name>\w+)\.call\(query="(?P<query>[^"]*)"\)'
CUSTOM_TOOL_CALL_PATTERN = re.compile(
    r"<function=(?P<function_name>[^}]+)>(?P<args>{.*?})"
)


def is_json(s):
    try:
        parsed = json.loads(s)
        # Return True for valid objects and not for ints, strings, etc
        return isinstance(parsed, dict)
    except json.JSONDecodeError:
        return False
    return True


class ToolUtils:

    @staticmethod
    def is_builtin_tool_call(message_body: str) -> bool:
        match = re.search(ToolUtils.BUILTIN_TOOL_PATTERN, message_body)
        return match is not None

    @staticmethod
    def maybe_extract_builtin_tool_call(message_body: str) -> Optional[Tuple[str, str]]:
        # Find the first match in the text
        match = re.search(BUILTIN_TOOL_PATTERN, message_body)

        # Check if a match is found and return it
        if match:
            tool_name = match.group("tool_name")
            query = match.group("query")
            return tool_name, query
        else:
            return None

    @staticmethod
    def maybe_extract_custom_tool_call(message_body: str) -> Optional[Tuple[str, str]]:
        # NOTE: Custom function too calls are still experimental
        # Sometimes, response is of the form
        # {"type": "function", "name": "function_name", "parameters": {...}
        # and some times
        # <function=function_name>(parameters)</function>

        # Find the first match in the text
        match = re.search(CUSTOM_TOOL_CALL_PATTERN, message_body)
        if match:
            tool_name = match.group("function_name")
            query = match.group("args")
            try:
                return tool_name, json.loads(query.replace("'", '"'))
            except Exception as e:
                print(
                    "Exception while parsing json query for custom tool call", query, e
                )
        elif is_json(message_body):
            response = json.loads(message_body)
            if ("type" in response and response["type"] == "function") or (
                "name" in response
            ):
                function_name = response["name"]
                args = response["parameters"]
                return function_name, args
            else:
                return None
        else:
            return None

    @staticmethod
    def encode_tool_call(t: ToolCall, tool_prompt_format: ToolPromptFormat) -> str:
        if t.tool_name == BuiltinTool.brave_search:
            q = t.arguments["query"]
            return f'brave_search.call(query="{q}")'
        elif t.tool_name == BuiltinTool.wolfram_alpha:
            q = t.arguments["query"]
            return f'wolfram_alpha.call(query="{q}")'
        elif t.tool_name == BuiltinTool.photogen:
            q = t.arguments["query"]
            return f'photogen.call(query="{q}")'
        elif t.tool_name == BuiltinTool.code_interpreter:
            return t.arguments["code"]
        else:
            fname = t.tool_name

            if tool_prompt_format == ToolPromptFormat.json:
                return json.dumps(
                    {
                        "type": "function",
                        "name": fname,
                        "parameters": t.arguments,
                    }
                )
            elif tool_prompt_format == ToolPromptFormat.function_tag:
                args = json.dumps(t.arguments)
                return f"<function={fname}>{args}</function>"
