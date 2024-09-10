# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from enum import Enum
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, validator

from typing_extensions import Annotated
from ...datatypes import *  # noqa
from PIL import Image as PIL_Image

from ...schema_utils import json_schema_type


@json_schema_type
class Role(Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    ipython = "ipython"


@json_schema_type(
    schema={"type": "string", "format": "uri", "pattern": "^(https?://|file://|data:)"}
)
class URL(BaseModel):
    uri: str

    def __str__(self) -> str:
        return self.uri


@json_schema_type
class ImageMedia(BaseModel):
    image: PIL_Image.Image

    class Config:
        arbitrary_types_allowed = True

    # TODO: this does not work with pydantic v1 I think
    @validator("image", pre=True, always=True)
    @classmethod
    def check_image(cls, v):
        if not isinstance(v, PIL_Image.Image):
            raise ValueError("Not a valid image")
        return v


InterleavedTextMedia = Union[
    str,
    # Specific modalities can be placed here, but not generic attachments
    # since models don't consume them in a generic way
    ImageMedia,
    List[Union[str, ImageMedia]],
]


def interleaved_text_media_as_str(content: InterleavedTextMedia, sep: str = " ") -> str:
    def _process(c) -> str:
        if isinstance(c, str):
            return c
        else:
            return "<media>"

    if isinstance(content, list):
        return sep.join(_process(c) for c in content)
    else:
        return _process(content)


@json_schema_type
class BuiltinTool(Enum):
    brave_search = "brave_search"
    wolfram_alpha = "wolfram_alpha"
    photogen = "photogen"
    code_interpreter = "code_interpreter"


Primitive = Union[str, int, float, bool, None]
RecursiveType = Union[Primitive, List[Primitive], Dict[str, Primitive]]


@json_schema_type
class ToolCall(BaseModel):
    call_id: str
    tool_name: Union[BuiltinTool, str]
    arguments: Dict[str, RecursiveType]

    @validator("tool_name", pre=True)
    @classmethod
    def validate_field(cls, v):
        if isinstance(v, str):
            try:
                return BuiltinTool(v)
            except ValueError:
                return v
        return v


@json_schema_type
class ToolResponse(BaseModel):
    call_id: str
    tool_name: Union[BuiltinTool, str]
    content: InterleavedTextMedia

    @validator("tool_name", pre=True)
    @classmethod
    def validate_field(cls, v):
        if isinstance(v, str):
            try:
                return BuiltinTool(v)
            except ValueError:
                return v
        return v


@json_schema_type
class ToolParamDefinition(BaseModel):
    param_type: str
    description: Optional[str] = None
    required: Optional[bool] = True


@json_schema_type
class ToolDefinition(BaseModel):
    tool_name: Union[BuiltinTool, str]
    description: Optional[str] = None
    parameters: Optional[Dict[str, ToolParamDefinition]] = None

    @validator("tool_name", pre=True)
    @classmethod
    def validate_field(cls, v):
        if isinstance(v, str):
            try:
                return BuiltinTool(v)
            except ValueError:
                return v
        return v


@json_schema_type
class ToolChoice(Enum):
    auto = "auto"
    required = "required"


@json_schema_type
class ToolPromptFormat(Enum):
    """This Enum refers to the prompt format for calling custom / zero shot tools

    `json` --
        Refers to the json format for calling tools.
        The json format takes the form like
        {
            "type": "function",
            "function" : {
                "name": "function_name",
                "description": "function_description",
                "parameters": {...}
            }
        }

    `function_tag` --
        This is an example of how you could define
        your own user defined format for making tool calls.
        The function_tag format looks like this,
        <function=function_name>(parameters)</function>

    The detailed prompts for each of these formats are added to llama cli
    """

    json = "json"
    function_tag = "function_tag"


@json_schema_type
class UserMessage(BaseModel):
    role: Literal[Role.user.value] = Role.user.value
    content: InterleavedTextMedia
    context: Optional[InterleavedTextMedia] = None


@json_schema_type
class SystemMessage(BaseModel):
    role: Literal[Role.system.value] = Role.system.value
    content: InterleavedTextMedia


@json_schema_type
class ToolResponseMessage(BaseModel):
    role: Literal[Role.ipython.value] = Role.ipython.value
    # it was nice to re-use the ToolResponse type, but having all messages
    # have a `content` type makes things nicer too
    call_id: str
    tool_name: Union[BuiltinTool, str]
    content: InterleavedTextMedia


@json_schema_type
class StopReason(Enum):
    end_of_turn = "end_of_turn"
    end_of_message = "end_of_message"
    out_of_tokens = "out_of_tokens"


@json_schema_type
class TokenLogProbs(BaseModel):
    logprobs_by_token: Dict[str, float]


@json_schema_type
class CompletionMessage(BaseModel):
    role: Literal[Role.assistant.value] = Role.assistant.value
    content: InterleavedTextMedia
    stop_reason: StopReason
    tool_calls: List[ToolCall] = Field(default_factory=list)


Message = Annotated[
    Union[
        UserMessage,
        SystemMessage,
        ToolResponseMessage,
        CompletionMessage,
    ],
    Field(discriminator="role"),
]
