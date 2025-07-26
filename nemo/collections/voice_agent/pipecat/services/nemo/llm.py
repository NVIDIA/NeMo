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
from typing import AsyncGenerator, List

import torch
from loguru import logger
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
from transformers import AsyncTextIteratorStreamer, AutoModelForCausalLM, AutoTokenizer


class HuggingFaceLLMLocalService:
    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = "cuda:0",
        temperature=0.7,
        max_tokens=256,
        top_p=0.9,
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, device_map=device, torch_dtype=torch.bfloat16
        )  # type: AutoModelForCausalLM
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    async def generate_stream(
        self, messages: List[ChatCompletionMessageParam], **kwargs
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        # Convert messages to prompt format

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        logger.debug(f"LLM prompt: {prompt}")

        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)

        # Generate with streaming
        streamer = AsyncTextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": True,
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
        temperature=0.7,
        max_tokens=256,
        top_p=0.9,
        **kwargs,
    ):
        self.model = model
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        super().__init__(model=model, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        return HuggingFaceLLMLocalService(
            model=self.model,
            device=self.device,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )

    async def _process_context(self, context: OpenAILLMContext):
        """Process a context through the LLM and push text frames.

        Args:
            context (OpenAILLMContext): The context to process, containing messages
                and other information needed for the LLM interaction.
        """
        await self.push_frame(LLMFullResponseStartFrame())

        try:
            await self.start_ttfb_metrics()
            messages = context.get_messages()
            async for chunk in self._client.generate_stream(messages):
                if chunk.choices[0].delta.content:
                    await self.stop_ttfb_metrics()
                    text = chunk.choices[0].delta.content
                    frame = LLMTextFrame(text)
                    await self.push_frame(frame)
        except Exception as e:
            logger.error(f"Error in _process_context: {e}", exc_info=True)
            raise
        finally:
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
