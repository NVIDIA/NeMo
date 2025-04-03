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

# Follwoing are temporarily adapted and modified from:
# https://github.com/meta-llama/llama-stack/tree/main/llama_stack/models/llama

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import io
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
from PIL import Image as PIL_Image
from llama_stack.models.llama.datatypes import (
    BuiltinTool,
    RawContent,
    RawMediaItem,
    RawMessage,
    RawTextItem,
    Role,
    StopReason,
    ToolCall,
    ToolPromptFormat,
)
from llama_stack.models.llama.llama3.tool_utils import ToolUtils

from .args import VisionArgs
from .datatypes import LLMInput
from .preprocess import ResizeNormalizeImageTransform, VariableSizeImageTransform


def get_llama4_formatter(tokenizer):
    data = {
        "image_height": 336,
        "image_width": 336,
        "patch_height": 14,
        "patch_width": 14,
        "pixel_shuffle_ratio": 0.5
    }
    data = VisionArgs.preprocess_values(data)
    vision_args = VisionArgs(**data)
    return ChatFormat(
        tokenizer,
        vision_args,
    )


def role_str(role: Role) -> str:
    role_strs = {
        Role.user: "user",
        Role.system: "system",
        Role.tool: "ipython",  # special
        Role.assistant: "assistant",
    }
    return role_strs[role]


@dataclass
class TransformedImage:
    image_tiles: torch.Tensor
    # is the aspect ratio needed anywhere?
    aspect_ratio: Tuple[int, int]


def pil_RGBA2RGB(image: PIL_Image.Image, bg: Tuple[int, int, int] = (255, 255, 255)) -> PIL_Image.Image:
    if image.mode == "RGBA":
        image.load()  # for png.split()
        new_img = PIL_Image.new("RGB", image.size, bg)
        new_img.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        return new_img
    return image.convert("RGB")


class ChatFormat:
    possible_headers: Dict[Role, str]

    def __init__(
            self,
            tokenizer: Any,
            vision_args: Optional[VisionArgs] = None,
            max_num_chunks: int = 16,
    ):
        self.tokenizer = tokenizer
        self.vision_args = vision_args
        self.max_num_chunks = max_num_chunks

        self.possible_headers = {role: f"<|header_start|>{role_str(role)}<|header_end|>\n\n" for role in Role}

        self.image_transform = None
        self.dynamic_image_transform = None
        if vision_args:
            self.dynamic_image_transform = VariableSizeImageTransform(vision_args.image_size.width)
            self.image_transform = ResizeNormalizeImageTransform(
                vision_args.image_size.width, vision_args.image_size.height
            )

    def _encode_header(self, role: str) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.vocab["<|header_start|>"])

        # TODO: need to check if this is correct
        tokens.extend(self.tokenizer.encode("ipython" if role == "tool" else role, add_special_tokens=False))
        tokens.append(self.tokenizer.vocab["<|header_end|>"])
        tokens.extend(self.tokenizer.encode("\n\n", add_special_tokens=False))
        return tokens

    def encode_content(self, content: RawContent) -> LLMInput:
        tokens, images = self._encode_content(content, bos=True)
        return self._model_input_from_tokens_images(tokens, images)

    def _encode_image(
            self,
            transformed_image: TransformedImage,
    ) -> List[int]:
        assert self.vision_args is not None, "The model is not vision-enabled"

        image_tensor = transformed_image.image_tiles
        image_channels = image_tensor.shape[-3]
        image_height = image_tensor.shape[-2]
        image_width = image_tensor.shape[-1]
        image_chunks = image_tensor.view(-1, image_channels, image_height, image_width).shape[0]

        patch_height = self.vision_args.patch_size.height
        patch_width = self.vision_args.patch_size.width

        if image_height % patch_height != 0:
            raise ValueError(f"{image_height=} not divisible by {patch_height=}")
        if image_width % patch_width != 0:
            raise ValueError(f"{image_width=} not divisible by {patch_width=}")

        ds_ratio = int(round(1.0 / (self.vision_args.pixel_shuffle_ratio ** 2)))
        n_patches_per_chunk = int((image_height // patch_height) * (image_width // patch_width) // ds_ratio)

        image_ar = transformed_image.aspect_ratio
        tokens = [self.tokenizer.vocab["<|image_start|>"]]
        if image_chunks == 1:
            tokens += [self.tokenizer.vocab["<|image|>"]]
            tokens += [self.tokenizer.vocab["<|patch|>"]] * n_patches_per_chunk
            tokens += [self.tokenizer.vocab["<|image_end|>"]]
        else:
            ratio_h, ratio_w = image_ar
            for yy in range(ratio_h):
                for xx in range(ratio_w):
                    tokens += [self.tokenizer.vocab["<|patch|>"]] * n_patches_per_chunk
                    if xx < ratio_w - 1:
                        tokens.append(self.tokenizer.vocab["<|tile_x_separator|>"])

                tokens.append(self.tokenizer.vocab["<|tile_y_separator|>"])

            tokens += [self.tokenizer.vocab["<|image|>"]]
            tokens += [self.tokenizer.vocab["<|patch|>"]] * n_patches_per_chunk
            tokens += [self.tokenizer.vocab["<|image_end|>"]]

        return tokens

    def _encode_content(self, content: RawContent, bos: bool = False) -> Tuple[List[int], List[TransformedImage]]:
        tokens = []
        tranformed_images = []

        added_bos = False

        def _process(c):
            nonlocal added_bos, bos

            if isinstance(c, str) or isinstance(c, RawTextItem):
                if isinstance(c, RawTextItem):
                    c = c.text
                tokens.extend(self.tokenizer.encode(c, add_special_tokens=False if added_bos else bos))
                added_bos = True

            elif isinstance(c, RawMediaItem):
                if not self.vision_args:
                    raise ValueError("The model is not vision-enabled, but a media item was found")

                bos = False if added_bos else bos
                if bos:
                    tokens.append(self.tokenizer.vocab["<|begin_of_text|>"])
                    added_bos = True

                bytes_io = io.BytesIO(c.data) if isinstance(c.data, bytes) else c.data
                image = PIL_Image.open(bytes_io)
                image = pil_RGBA2RGB(image)
                image_tiles, ar = self.dynamic_image_transform(image, max_num_chunks=self.max_num_chunks)

                if image_tiles.shape[0] > 1:
                    image_global = self.image_transform(image)
                    image_global = image_global.unsqueeze(0)
                    image_combine = torch.cat((image_tiles, image_global), dim=0)
                    image_tiles = image_combine

                transformed_image = TransformedImage(image_tiles=image_tiles, aspect_ratio=ar)
                tokens.extend(self._encode_image(transformed_image))
                tranformed_images.append(transformed_image)

        if isinstance(content, list):
            for c in content:
                _process(c)
        else:
            _process(content)

        return tokens, tranformed_images

    def encode_message(
            self, message: RawMessage, tool_prompt_format: ToolPromptFormat
    ) -> Tuple[List[int], List[TransformedImage]]:
        tokens = self._encode_header(message.role)
        images = []

        def _process_content(c):
            toks, imgs = self._encode_content(c)
            tokens.extend(toks)
            images.extend(imgs)

        _process_content(message.content)

        if message.role == "user" and message.context is not None:
            # This is RAG context; why is it here in the chat format? I don't think
            # this is needed and can be moved upwards
            _process_content("\n\n")
            _process_content(message.context)

        if message.role == "assistant":
            for t in message.tool_calls:
                content = ToolUtils.encode_tool_call(t, tool_prompt_format)
                _process_content(content)

        eom = False
        if message.role == "assistant":
            eom = message.stop_reason == StopReason.end_of_message

        tokens.append(self.tokenizer.vocab["<|eom|>" if eom else "<|eot|>"])
        return tokens, images

    def encode_dialog_prompt(
            self,
            messages: List[RawMessage],
            tool_prompt_format: ToolPromptFormat = ToolPromptFormat.json,
    ) -> LLMInput:
        tokens = []
        images = []
        tokens.append(self.tokenizer.vocab["<|begin_of_text|>"])
        for message in messages:
            toks, imgs = self.encode_message(message, tool_prompt_format)
            tokens.extend(toks)
            images.extend(imgs)

        # Add the start of an assistant message for the model to complete.
        tokens.extend(self._encode_header("assistant"))

        return self._model_input_from_tokens_images(tokens, images)

    def decode_assistant_message(self, tokens: List[int], stop_reason: StopReason) -> RawMessage:
        content = self.tokenizer.decode(tokens)

        return self.decode_assistant_message_from_content(content, stop_reason)

    def decode_assistant_message_from_content(self, content: str, stop_reason: StopReason) -> RawMessage:
        content = content.strip(" ")
        header_str = self.possible_headers[Role.assistant]
        if content.startswith(header_str):
            content = content[len(header_str) :]

        if content.endswith("<|eot|>"):
            content = content[: -len("<|eot|>")]
            stop_reason = StopReason.end_of_turn
        elif content.endswith("<|eom|>"):
            content = content[: -len("<|eom|>")]
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

        return RawMessage(
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
        )

    def _model_input_from_tokens_images(self, tokens: List[int], images: List[TransformedImage]) -> LLMInput:
        return LLMInput(
            tokens=tokens,
            images=[x.image_tiles for x in images] if len(images) > 0 else None,
        )
