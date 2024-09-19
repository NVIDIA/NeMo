# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, validator

from typing_extensions import Annotated

import base64
import re
from io import BytesIO

from PIL import Image as PIL_Image

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Type, TypeVar, Union

# Borrowed from https://github.com/hunyadi/strong_typing/blob/master/strong_typing/core.py


class JsonObject:
    "Placeholder type for an unrestricted JSON object."


class JsonArray:
    "Placeholder type for an unrestricted JSON array."


# a JSON type with possible `null` values
JsonType = Union[
    None,
    bool,
    int,
    float,
    str,
    Dict[str, "JsonType"],
    List["JsonType"],
]

# a JSON type that cannot contain `null` values
StrictJsonType = Union[
    bool,
    int,
    float,
    str,
    Dict[str, "StrictJsonType"],
    List["StrictJsonType"],
]

# a meta-type that captures the object type in a JSON schema
Schema = Dict[str, JsonType]


T = TypeVar("T")


def register_schema(
    data_type: T,
    schema: Optional[Schema] = None,
    name: Optional[str] = None,
    examples: Optional[List[JsonType]] = None,
) -> T:
    """
    Associates a type with a JSON schema definition.

    :param data_type: The type to associate with a JSON schema.
    :param schema: The schema to associate the type with. Derived automatically if omitted.
    :param name: The name used for looking uo the type. Determined automatically if omitted.
    :returns: The input type.
    """
    return data_type


def json_schema_type(
    cls: Optional[Type[T]] = None,
    *,
    schema: Optional[Schema] = None,
    examples: Optional[List[JsonType]] = None,
) -> Union[Type[T], Callable[[Type[T]], Type[T]]]:
    """Decorator to add user-defined schema definition to a class."""

    def wrap(cls: Type[T]) -> Type[T]:
        return register_schema(cls, schema, examples=examples)

    # see if decorator is used as @json_schema_type or @json_schema_type()
    if cls is None:
        # called with parentheses
        return wrap
    else:
        # called as @json_schema_type without parentheses
        return wrap(cls)


register_schema(JsonObject, name="JsonObject")
register_schema(JsonArray, name="JsonArray")
register_schema(JsonType, name="JsonType")
register_schema(StrictJsonType, name="StrictJsonType")


@dataclass
class WebMethod:
    route: Optional[str] = None
    public: bool = False
    request_examples: Optional[List[Any]] = None
    response_examples: Optional[List[Any]] = None
    method: Optional[str] = None


def webmethod(
    route: Optional[str] = None,
    method: Optional[str] = None,
    public: Optional[bool] = False,
    request_examples: Optional[List[Any]] = None,
    response_examples: Optional[List[Any]] = None,
) -> Callable[[T], T]:
    """
    Decorator that supplies additional metadata to an endpoint operation function.

    :param route: The URL path pattern associated with this operation which path parameters are substituted into.
    :param public: True if the operation can be invoked without prior authentication.
    :param request_examples: Sample requests that the operation might take. Pass a list of objects, not JSON.
    :param response_examples: Sample responses that the operation might produce. Pass a list of objects, not JSON.
    """

    def wrap(cls: T) -> T:
        cls.__webmethod__ = WebMethod(
            route=route,
            method=method,
            public=public or False,
            request_examples=request_examples,
            response_examples=response_examples,
        )
        return cls

    return wrap

@json_schema_type
class SamplingStrategy(Enum):
    greedy = "greedy"
    top_p = "top_p"
    top_k = "top_k"


@json_schema_type
class SamplingParams(BaseModel):
    strategy: SamplingStrategy = SamplingStrategy.greedy

    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 0
    max_tokens: Optional[int] = 0
    repetition_penalty: Optional[float] = 1.0


@json_schema_type(
    schema={
        "description": """
The format in which weights are specified. This does not necessarily
always equal what quantization is desired at runtime since there
can be on-the-fly conversions done.
""",
    }
)
class CheckpointQuantizationFormat(Enum):
    # default format
    bf16 = "bf16"

    # used for enabling fp8_rowwise inference, some weights are bf16
    fp8_mixed = "fp8-mixed"

    int8 = "int8"

    int4 = "int4"


@json_schema_type
class ModelFamily(Enum):
    llama2 = "llama2"
    llama3 = "llama3"
    llama3_1 = "llama3_1"
    llama3_2 = "llama3_2"
    safety = "safety"


@json_schema_type
class CoreModelId(Enum):
    """Each of these models is a unique "SKU". These root models can be served in various garbs (especially by quantizing them)"""

    # Llama 2 family
    llama2_7b = "Llama-2-7b"
    llama2_13b = "Llama-2-13b"
    llama2_70b = "Llama-2-70b"
    llama2_7b_chat = "Llama-2-7b-chat"
    llama2_13b_chat = "Llama-2-13b-chat"
    llama2_70b_chat = "Llama-2-70b-chat"

    # Llama 3 family
    llama3_8b = "Llama-3-8B"
    llama3_70b = "Llama-3-70B"
    llama3_8b_instruct = "Llama-3-8B-Instruct"
    llama3_70b_instruct = "Llama-3-70B-Instruct"

    # Llama 3.1 family
    llama3_1_8b = "Llama3.1-8B"
    llama3_1_70b = "Llama3.1-70B"
    llama3_1_405b = "Llama3.1-405B"
    llama3_1_8b_instruct = "Llama3.1-8B-Instruct"
    llama3_1_70b_instruct = "Llama3.1-70B-Instruct"
    llama3_1_405b_instruct = "Llama3.1-405B-Instruct"

    # Llama 3.2 family
    llama3_2_1b = "Llama3.2-1B"
    llama3_2_3b = "Llama3.2-3B"
    llama3_2_1b_instruct = "Llama3.2-1B-Instruct"
    llama3_2_3b_instruct = "Llama3.2-3B-Instruct"
    llama3_2_11b_vision = "Llama3.2-11B-Vision"
    llama3_2_90b_vision = "Llama3.2-90B-Vision"
    llama3_2_11b_vision_instruct = "Llama3.2-11B-Vision-Instruct"
    llama3_2_90b_vision_instruct = "Llama3.2-90B-Vision-Instruct"

    # Safety models
    llama_guard_3_8b = "Llama-Guard-3-8B"
    prompt_guard_86m = "Prompt-Guard-86M"
    llama_guard_2_8b = "Llama-Guard-2-8B"
    llama_guard_3_11b_vision = "Llama-Guard-3-11B-Vision"
    llama_guard_3_1b = "Llama-Guard-3-1B"


def model_family(model_id) -> ModelFamily:
    if model_id in [
        CoreModelId.llama2_7b,
        CoreModelId.llama2_13b,
        CoreModelId.llama2_70b,
        CoreModelId.llama2_7b_chat,
        CoreModelId.llama2_13b_chat,
        CoreModelId.llama2_70b_chat,
    ]:
        return ModelFamily.llama2
    elif model_id in [
        CoreModelId.llama3_8b,
        CoreModelId.llama3_70b,
        CoreModelId.llama3_8b_instruct,
        CoreModelId.llama3_70b_instruct,
    ]:
        return ModelFamily.llama3
    elif model_id in [
        CoreModelId.llama3_1_8b,
        CoreModelId.llama3_1_70b,
        CoreModelId.llama3_1_405b,
        CoreModelId.llama3_1_8b_instruct,
        CoreModelId.llama3_1_70b_instruct,
        CoreModelId.llama3_1_405b_instruct,
    ]:
        return ModelFamily.llama3_1
    elif model_id in [
        CoreModelId.llama3_2_1b,
        CoreModelId.llama3_2_3b,
        CoreModelId.llama3_2_1b_instruct,
        CoreModelId.llama3_2_3b_instruct,
        CoreModelId.llama3_2_11b_vision,
        CoreModelId.llama3_2_90b_vision,
        CoreModelId.llama3_2_11b_vision_instruct,
        CoreModelId.llama3_2_90b_vision_instruct,
    ]:
        return ModelFamily.llama3_2
    elif model_id in [
        CoreModelId.llama_guard_3_8b,
        CoreModelId.prompt_guard_86m,
        CoreModelId.llama_guard_2_8b,
        CoreModelId.llama_guard_3_11b_vision,
        CoreModelId.llama_guard_3_1b,
    ]:
        return ModelFamily.safety
    else:
        raise ValueError(f"Unknown model family for {CoreModelId}")


@json_schema_type(
    schema={
        "description": "The model family and SKU of the model along with other parameters corresponding to the model."
    }
)
class Model(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    core_model_id: CoreModelId
    is_default_variant: bool

    @property
    def model_family(self) -> ModelFamily:
        return model_family(self.core_model_id)

    # Featured models are shown in the non-exhaustive model list
    @property
    def is_featured(self) -> bool:
        return self.model_family in [
            ModelFamily.llama3_1,
            ModelFamily.llama3_2,
            ModelFamily.safety,
        ]

    @property
    def max_seq_length(self) -> int:
        if self.model_family == ModelFamily.llama2:
            return 4096
        elif self.core_model_id == CoreModelId.llama_guard_2_8b:
            return 4096
        elif self.model_family == ModelFamily.llama3:
            return 8192
        elif self.model_family == ModelFamily.llama3_1:
            return 131072
        elif self.model_family == ModelFamily.llama3_2:
            return 131072
        elif self.core_model_id in [
            CoreModelId.llama_guard_3_8b,
            CoreModelId.prompt_guard_86m,
            CoreModelId.llama_guard_3_11b_vision,
            CoreModelId.llama_guard_3_1b,
        ]:
            return 131072
        else:
            raise ValueError(f"Unknown max_seq_len for {self.core_model_id}")

    # The variant is a string representation of other parameters which helps
    # uniquely identify the model. this typically includes the quantization
    # format, model parallel size, etc.
    @property
    def variant(self) -> str:
        parts = [
            self.quantization_format.value,
            f"mp{self.pth_file_count}",
        ]

        return "-".join(parts)

    # The SKU is uniquely identified by (model_id, variant) combo
    def descriptor(self, shorten_default_variant: bool = True) -> str:
        if shorten_default_variant and self.is_default_variant:
            return self.core_model_id.value

        return f"{self.core_model_id.value}:{self.variant}"

    description_markdown: str
    huggingface_repo: Optional[str] = None
    quantization_format: CheckpointQuantizationFormat = (
        CheckpointQuantizationFormat.bf16
    )
    recommended_sampling_params: Optional[SamplingParams] = None
    model_args: Dict[str, Any]
    pth_file_count: int
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @property
    def is_instruct_model(self) -> bool:
        return "instruct" in self.id.name

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
    image: Union[PIL_Image.Image, URL]

    class Config:
        arbitrary_types_allowed = True


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


def interleaved_text_media_localize(
    content: InterleavedTextMedia,
) -> InterleavedTextMedia:
    def _localize_single(c: str | ImageMedia) -> str | ImageMedia:
        if isinstance(c, ImageMedia):
            # load image and return PIL version
            img = c.image
            if isinstance(img, URL):
                if img.uri.startswith("file://"):
                    img = PIL_Image.open(img.uri[len("file://") :]).convert("RGB")
                elif img.uri.startswith("data"):
                    match = re.match(r"data:image/(\w+);base64,(.+)", img.uri)
                    if not match:
                        raise ValueError("Invalid data URL format")
                    image_type, image_data = match.groups()
                    image_data = base64.b64decode(image_data)
                    img = PIL_Image.open(BytesIO(image_data))
                else:
                    raise ValueError("Unsupported URL type")
            return ImageMedia(image=img)
        else:
            return c

    if isinstance(content, list):
        return [_localize_single(c) for c in content]
    else:
        return _localize_single(content)


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
