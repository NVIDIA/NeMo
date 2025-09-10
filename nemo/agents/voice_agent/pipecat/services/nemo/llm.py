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

import time
import uuid
from threading import Thread
from typing import AsyncGenerator, List, Mapping, Optional

from jinja2.exceptions import TemplateError
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from openai import AsyncStream, BadRequestError
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
from transformers import AsyncTextIteratorStreamer, AutoModelForCausalLM, AutoTokenizer

DEFAULT_GENERATION_KWARGS = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}


class LLMUtilsMixin:
    def _maybe_add_user_message(self, messages: List[ChatCompletionMessageParam]) -> List[ChatCompletionMessageParam]:
        """
        Some LLMs like "nvidia/Llama-3.1-Nemotron-Nano-8B-v1" requires a user turn after the system prompt, this function is used to add a dummy user turn if the system prompt is followed by an assistant turn.
        """
        if len(messages) > 1 and messages[0]["role"] == "system" and messages[1]["role"] == "assistant":
            message = {"role": "user", "content": "Hi"}
            messages.insert(1, message)
        elif len(messages) == 1 and messages[0]["role"] == "system":
            messages.append({"role": "user", "content": "Hi"})
        return messages

    def _maybe_merge_consecutive_turns(
        self, messages: List[ChatCompletionMessageParam]
    ) -> List[ChatCompletionMessageParam]:
        """
        Merge consecutive turns of the same role into a single turn, since some LLMs like "nvidia/Llama-3.1-Nemotron-Nano-8B-v1" do not support consecutive turns of the same role.
        """
        if not messages:
            return messages

        merged_messages = []
        current_role = None
        current_content = ""

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == current_role:
                # Merge with previous message of same role
                current_content += "; " + content
            else:
                # Save previous message if exists
                if current_role is not None:
                    merged_messages.append({"role": current_role, "content": current_content})

                # Start new message
                current_role = role
                current_content = content

        # Add the last message
        if current_role is not None:
            merged_messages.append({"role": current_role, "content": current_content})

        return merged_messages


class HuggingFaceLLMLocalService(LLMUtilsMixin):
    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        thinking_budget: int = 0,
        generation_kwargs: dict = None,
        apply_chat_template_kwargs: dict = None,
    ):
        self.device = device
        self.dtype = dtype
        self.thinking_budget = thinking_budget
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, device_map=device, torch_dtype=dtype, trust_remote_code=True
        )  # type: AutoModelForCausalLM

        self.generation_kwargs = generation_kwargs if generation_kwargs else DEFAULT_GENERATION_KWARGS
        logger.debug(f"LLM generation kwargs: {self.generation_kwargs}")

        self.apply_chat_template_kwargs = apply_chat_template_kwargs if apply_chat_template_kwargs else {}
        if "tokenize" in self.apply_chat_template_kwargs:
            if self.apply_chat_template_kwargs["tokenize"] is not False:
                logger.warning(
                    f"Found `tokenize=True` in apply_chat_template_kwargs, it will be ignored and forced to `False`"
                )
            self.apply_chat_template_kwargs.pop("tokenize")

        logger.debug(f"LLM apply_chat_template kwargs: {self.apply_chat_template_kwargs}")

    def _apply_chat_template(self, messages: List[ChatCompletionMessageParam]) -> str:
        """
        Apply the chat template to the messages.
        """
        return self.tokenizer.apply_chat_template(messages, tokenize=False, **self.apply_chat_template_kwargs)

    def _get_prompt_from_messages(self, messages: List[ChatCompletionMessageParam]) -> str:
        """
        Get the formatted prompt from the conversation history messages.
        This function also tries to fix the messages if the LLM cannot handle consecutive turns of the same role,
        or requires a user turn after the system prompt.
        """
        try:
            prompt = self._apply_chat_template(messages)
            return prompt
        except TemplateError as e:
            logger.warning(f"Got TemplateError: {e}.")

        logger.debug(f"Input LLM messages: {messages}")
        if len(messages) > 1 and messages[0]["role"] == "system" and messages[1]["role"] == "assistant":
            logger.warning("Trying to fix by adding dummy user message after system prompt...")
            try:
                messages = self._maybe_add_user_message(messages)
                logger.debug(f"LLM messages after adding dummy user message: {messages}")
                prompt = self._apply_chat_template(messages)
                return prompt
            except TemplateError as e:
                logger.warning(f"Got TemplateError: {e}. Trying to fix by merging consecutive turns if possible.")

        try:
            new_messages = self._maybe_merge_consecutive_turns(messages)
            logger.debug(f"LLM messages after merging consecutive user turns: {new_messages}")
            prompt = self._apply_chat_template(new_messages)
            # Update the messages in place if successful
            messages.clear()
            messages.extend(new_messages)
            return prompt
        except Exception as e:
            logger.warning(f"Got Exception: {e}, messages: {messages}")
            raise e

    async def generate_stream(
        self, messages: List[ChatCompletionMessageParam], **kwargs
    ) -> AsyncGenerator[ChatCompletionChunk, None]:

        # Convert messages to prompt format
        prompt = self._get_prompt_from_messages(messages)

        logger.debug(f"LLM prompt: {prompt}")

        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)

        # Generate with streaming
        streamer = AsyncTextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            **self.generation_kwargs,
        }

        # Start generation in background
        thread = Thread(
            target=self.model.generate,
            kwargs=generation_kwargs,
        )
        thread.start()

        # Stream the output
        async for text in streamer:
            # logger.debug(f"Streamer yielded text: {text}")
            chunk = ChatCompletionChunk(
                id="hf-" + str(uuid.uuid4()),
                choices=[{"delta": {"content": text}, "finish_reason": None, "index": 0}],
                created=int(time.time()),
                model=self.model.config._name_or_path,
                object="chat.completion.chunk",
            )
            yield chunk


class HuggingFaceLLMService(OpenAILLMService):
    def __init__(
        self,
        *,
        model: str = "google/gemma-7b-it",
        device: str = "cuda",
        dtype: str = "bfloat16",
        thinking_budget: int = 0,
        generation_kwargs: dict = None,
        apply_chat_template_kwargs: dict = None,
        **kwargs,
    ):
        self._model_name = model
        self._device = device
        self._dtype = dtype
        self._thinking_budget = thinking_budget
        self._generation_kwargs = generation_kwargs if generation_kwargs is not None else DEFAULT_GENERATION_KWARGS
        self._apply_chat_template_kwargs = apply_chat_template_kwargs if apply_chat_template_kwargs is not None else {}
        super().__init__(model=model, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        return HuggingFaceLLMLocalService(
            model=self._model_name,
            device=self._device,
            dtype=self._dtype,
            thinking_budget=self._thinking_budget,
            generation_kwargs=self._generation_kwargs,
            apply_chat_template_kwargs=self._apply_chat_template_kwargs,
        )

    async def _process_context(self, context: OpenAILLMContext):
        """Process a context through the LLM and push text frames.

        Args:
            context (OpenAILLMContext): The context to process, containing messages
                and other information needed for the LLM interaction.
        """
        await self.push_frame(LLMFullResponseStartFrame())
        cumulative_text = ""
        try:
            await self.start_ttfb_metrics()
            messages = context.get_messages()
            async for chunk in self._client.generate_stream(messages):
                if chunk.choices[0].delta.content:
                    await self.stop_ttfb_metrics()
                    text = chunk.choices[0].delta.content
                    cumulative_text += text
                    frame = LLMTextFrame(text)
                    await self.push_frame(frame)
        except Exception as e:
            logger.error(f"Error in _process_context: {e}", exc_info=True)
            raise
        finally:
            cumulative_text = " ".join(cumulative_text.split()).strip()
            if not cumulative_text:
                logger.warning(f"LLM response is empty for context: {context}")
            await self.push_frame(LLMFullResponseEndFrame())

    async def get_chat_completions(
        self, context: OpenAILLMContext, messages: List[ChatCompletionMessageParam]
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Create a streaming chat completion using HuggingFace model.

        Args:
            context (OpenAILLMContext): The context object containing tools configuration
                and other settings for the chat completion.
            messages (List[ChatCompletionMessageParam]): The list of messages comprising
                the conversation history and current request.

        Returns:
            AsyncGenerator[ChatCompletionChunk]: A streaming response of chat completion
                chunks that can be processed asynchronously.
        """
        params = {
            "max_tokens": self._settings["max_tokens"],
            "temperature": self._settings["temperature"],
            "top_p": self._settings["top_p"],
        }
        params.update(self._settings["extra"])

        return self._client.generate_stream(messages, **params)


class VLLMService(OpenAILLMService, LLMUtilsMixin):
    def __init__(
        self,
        *,
        model: str,
        api_key="None",
        base_url="http://localhost:8000/v1",
        organization="None",
        project="None",
        default_headers: Optional[Mapping[str, str]] = None,
        params: Optional[OpenAILLMService.InputParams] = None,
        thinking_budget: int = 0,
        **kwargs,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            default_headers=default_headers,
            params=params,
            **kwargs,
        )
        self._thinking_budget = thinking_budget
        # TODO: handle thinking budget
        logger.info(
            f"VLLMService initialized with model: {model}, api_key: {api_key}, base_url: {base_url}, params: {params}, thinking_budget: {thinking_budget}"
        )

    async def get_chat_completions(
        self, context: OpenAILLMContext, messages: List[ChatCompletionMessageParam]
    ) -> AsyncStream[ChatCompletionChunk]:
        """Get streaming chat completions from OpenAI API.

        Args:
            context: The LLM context containing tools and configuration.
            messages: List of chat completion messages to send.

        Returns:
            Async stream of chat completion chunks.
        """

        params = {
            "model": self.model_name,
            "stream": True,
            "messages": messages,
            "tools": context.tools,
            "tool_choice": context.tool_choice,
            "stream_options": {"include_usage": True},
            "frequency_penalty": self._settings["frequency_penalty"],
            "presence_penalty": self._settings["presence_penalty"],
            "seed": self._settings["seed"],
            "temperature": self._settings["temperature"],
            "top_p": self._settings["top_p"],
            "max_tokens": self._settings["max_tokens"],
            "max_completion_tokens": self._settings["max_completion_tokens"],
        }

        params.update(self._settings["extra"])
        chunks = await self._get_response_from_client(messages, params)
        return chunks

    async def _get_response_from_client(
        self, messages: List[ChatCompletionMessageParam], params: dict
    ) -> AsyncStream[ChatCompletionChunk]:
        try:
            chunks = await self._client.chat.completions.create(**params)
        except BadRequestError as e:
            logger.warning(
                f"Error in get_chat_completions: {e}, trying to fix by adding dummy user message and merging consecutive turns if possible."
            )
            logger.debug(f"LLM messages before fixing: {messages}")
            messages = self._maybe_add_user_message(messages)
            messages = self._maybe_merge_consecutive_turns(messages)
            logger.debug(f"LLM messages after fixing: {messages}")
            params["messages"] = messages
            chunks = await self._client.chat.completions.create(**params)
        return chunks


def get_llm_service_from_config(config: DictConfig) -> OpenAILLMService:
    backend = config.type
    assert backend in ["hf", "vllm"], f"Invalid backend: {backend}, only `hf` and `vllm` are supported."
    if backend == "hf":
        llm_model = config.model
        llm_device = config.device
        llm_dtype = config.dtype
        llm_generation_kwargs = config.get("generation_kwargs", {})
        if llm_generation_kwargs is not None:
            llm_generation_kwargs = OmegaConf.to_container(llm_generation_kwargs)
        llm_apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", None)
        if llm_apply_chat_template_kwargs is not None:
            llm_apply_chat_template_kwargs = OmegaConf.to_container(llm_apply_chat_template_kwargs)
        llm_thinking_budget = config.get("thinking_budget", 0)
        return HuggingFaceLLMService(
            model=llm_model,
            device=llm_device,
            dtype=llm_dtype,
            generation_kwargs=llm_generation_kwargs,
            apply_chat_template_kwargs=llm_apply_chat_template_kwargs,
            thinking_budget=llm_thinking_budget,
        )
    elif backend == "vllm":
        llm_model = config.get("model", "vllm_server")
        llm_api_key = config.get("api_key", "None")
        llm_base_url = config.get("base_url", "http://localhost:8000/v1")
        llm_organization = config.get("organization", "None")
        llm_project = config.get("project", "None")
        llm_default_headers = config.get("default_headers", None)
        llm_params = config.get("params", None)
        if llm_params is not None:
            # cast into OpenAILLMService.InputParams object
            llm_params = OmegaConf.to_container(llm_params, resolve=True)
            extra = llm_params.get("extra", None)
            # ensure extra is a dictionary
            if extra is None:
                llm_params["extra"] = {}
            elif not isinstance(extra, dict):
                raise ValueError(f"extra must be a dictionary, got {type(extra)}")
            llm_params = OpenAILLMService.InputParams(**llm_params)
        else:
            llm_params = OpenAILLMService.InputParams()
        llm_thinking_budget = config.get("thinking_budget", 0)
        return VLLMService(
            model=llm_model,
            api_key=llm_api_key,
            base_url=llm_base_url,
            organization=llm_organization,
            project=llm_project,
            default_headers=llm_default_headers,
            params=llm_params,
            thinking_budget=llm_thinking_budget,
        )
    else:
        raise ValueError(f"Invalid LLM backend: {backend}")
