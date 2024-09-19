# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import uuid

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .tokenizer import Tokenizer
from .datatypes import *  # noqa: F403
from PIL import Image as PIL_Image

from .tool_utils import ToolUtils


@dataclass
class VisionInput:
    mask: List[List[int]]
    images: List[PIL_Image.Image]


@dataclass
class ModelInput:
    tokens: List[int]
    vision: Optional[VisionInput] = None


class ChatFormat:
    possible_headers: Dict[Role, str]

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.possible_headers = {
            role: f"<|start_header_id|>{role.value}<|end_header_id|>\n\n"
            for role in Role
        }
        self.vision_token = self.tokenizer.special_tokens["<|image|>"]

    def _encode_header(self, role: str) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(role, bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_content(self, content: InterleavedTextMedia) -> ModelInput:
        tokens, images = self._encode_content(content, bos=True)
        return self._model_input_from_tokens_images(tokens, images)

    def _encode_content(
        self, content: InterleavedTextMedia, bos: bool = False
    ) -> Tuple[List[int], List[PIL_Image.Image]]:
        tokens = []
        images = []

        added_bos = False

        def _process(c):
            nonlocal added_bos

            if isinstance(c, str):
                tokens.extend(
                    self.tokenizer.encode(c, bos=False if added_bos else bos, eos=False)
                )
                added_bos = True
            elif isinstance(c, ImageMedia):
                tokens.append(self.vision_token)
                cc = interleaved_text_media_localize(c)
                images.append(cc.image)

        if isinstance(content, str):
            _process(content)
        elif isinstance(content, list):
            for c in content:
                _process(c)

        return tokens, images

    def encode_message(
        self, message: Message, tool_prompt_format: ToolPromptFormat
    ) -> Tuple[List[int], List[PIL_Image.Image]]:
        tokens = self._encode_header(message.role)
        images = []

        def _process_content(c):
            toks, imgs = self._encode_content(c)
            tokens.extend(toks)
            images.extend(imgs)

        if isinstance(message, CompletionMessage) and len(message.tool_calls) > 0:
            tokens.append(self.tokenizer.special_tokens["<|python_tag|>"])

        _process_content(message.content)

        if isinstance(message, UserMessage) and message.context is not None:
            _process_content("\n\n")
            _process_content(message.context)

        if isinstance(message, CompletionMessage):
            for t in message.tool_calls:
                content = ToolUtils.encode_tool_call(t, tool_prompt_format)
                _process_content(content)

        eom = False
        if isinstance(message, CompletionMessage):
            eom = message.stop_reason == StopReason.end_of_message

        tokens.append(
            self.tokenizer.special_tokens["<|eom_id|>" if eom else "<|eot_id|>"]
        )
        return tokens, images

    def encode_dialog_prompt(
        self,
        messages: List[Message],
        tool_prompt_format: ToolPromptFormat = ToolPromptFormat.json,
    ) -> ModelInput:
        tokens = []
        images = []
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        for message in messages:
            toks, imgs = self.encode_message(message, tool_prompt_format)
            tokens.extend(toks)
            images.extend(imgs)

        # Add the start of an assistant message for the model to complete.
        tokens.extend(self._encode_header(Role.assistant.value))

        return self._model_input_from_tokens_images(tokens, images)

    # TODO(this should be generic, not only for assistant messages)
    def decode_assistant_message(
        self, tokens: List[int], stop_reason: StopReason
    ) -> CompletionMessage:
        content = self.tokenizer.decode(tokens)

        return self.decode_assistant_message_from_content(content, stop_reason)

    def decode_assistant_message_from_content(
        self, content: str, stop_reason: StopReason
    ) -> CompletionMessage:
        content = content.strip(" ")
        header_str = self.possible_headers[Role.assistant]
        if content.startswith(header_str):
            content = content[len(header_str) :]

        ipython = content.startswith("<|python_tag|>")
        if ipython:
            content = content[len("<|python_tag|>") :]

        if content.endswith("<|eot_id|>"):
            content = content[: -len("<|eot_id|>")]
            stop_reason = StopReason.end_of_turn
        elif content.endswith("<|eom_id|>"):
            content = content[: -len("<|eom_id|>")]
            stop_reason = StopReason.end_of_message

        tool_name = None
        tool_arguments = {}

        custom_tool_info = ToolUtils.maybe_extract_custom_tool_call(content)
        if custom_tool_info is not None:
            tool_name, tool_arguments = custom_tool_info
            # Sometimes when agent has custom tools alongside builin tools
            # Agent responds for builtin tool calls in the format of the custom tools
            # This code tries to handle that case
            if tool_name in BuiltinTool.__members__:
                tool_name = BuiltinTool[tool_name]
                tool_arguments = {
                    "query": list(tool_arguments.values())[0],
                }
        else:
            builtin_tool_info = ToolUtils.maybe_extract_builtin_tool_call(content)
            if builtin_tool_info is not None:
                tool_name, query = builtin_tool_info
                tool_arguments = {
                    "query": query,
                }
                if tool_name in BuiltinTool.__members__:
                    tool_name = BuiltinTool[tool_name]
            elif ipython:
                tool_name = BuiltinTool.code_interpreter
                tool_arguments = {
                    "code": content,
                }

        tool_calls = []
        if tool_name is not None and tool_arguments is not None:
            call_id = str(uuid.uuid4())
            tool_calls.append(
                ToolCall(
                    call_id=call_id,
                    tool_name=tool_name,
                    arguments=tool_arguments,
                )
            )
            content = ""

        return CompletionMessage(
            content=content,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
        )

    def _model_input_from_tokens_images(
        self, tokens: List[int], images: List[PIL_Image.Image]
    ) -> ModelInput:
        vision_input = None
        if len(images) > 0:
            vision_input = VisionInput(
                mask=create_vision_mask(tokens, self.vision_token),
                images=images,
            )

        return ModelInput(
            tokens=[
                128256 if token == self.vision_token else token for token in tokens
            ],
            vision=vision_input,
        )


def create_vision_mask(
    tokens: List[int],
    vision_token: int,
) -> List[List[int]]:
    vision_token_locations = [
        i for i, token in enumerate(tokens) if token == vision_token
    ]
    if len(vision_token_locations) == 0:
        return []

    if len(vision_token_locations) == 1:
        # only one image present, unmask until end of sequence
        return [[vision_token_locations[0], -1]]
    vision_masks = [
        [loc1, loc2]
        for loc1, loc2 in zip(vision_token_locations[:-1], vision_token_locations[1:])
    ]
    # last image will attend to all subsequent text
    vision_masks.append([vision_token_locations[-1], len(tokens)])

    # if there are two or more consecutive vision tokens,
    # they should all attend to all subsequent
    # text present
    last_mask_end = vision_masks[-1][1]
    for vision_mask in vision_masks[::-1]:
        if vision_mask[0] == vision_mask[1] - 1:
            vision_mask[1] = last_mask_end
        last_mask_end = vision_mask[1]
    return vision_masks
