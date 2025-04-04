# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, TypedDict

import torch

from cosmos1.models.autoregressive.model import AutoRegressiveModel
from cosmos1.models.autoregressive.tokenizer.image_text_tokenizer import ImageTextTokenizer
from cosmos1.models.autoregressive.tokenizer.text_tokenizer import TextTokenizer


class ChatPrediction(TypedDict, total=False):
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


def chat_completion(
    model: AutoRegressiveModel,
    dialogs: List,
    seed: int = None,
    temperature: float = 0.01,
    top_k: int = None,
    top_p: float = None,
    max_gen_len: Optional[int] = None,
    num_gen_seq: int = 1,
    logprobs: bool = False,
    generation_prefix: str = "",
    compile_sampling: bool = False,
    compile_prefill: bool = False,
    stop_tokens=None,
    verbose: bool = False,
) -> List[ChatPrediction]:
    """
    Generate assistant responses for a list of conversational dialogs using the language generation model.

    Args:
        model (AutoRegressiveModel): The language generation model.
        dialogs (List): List of conversational dialogs, where each dialog is a list of messages.
            NOTE if you are using a VLM, all dialogs must either all have images ("image" field) or all be pure text.
        temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.01.
        top_k (int, optional): Top-k probability threshold for nucleus sampling. Defaults to None. If not None, top-p sampling is ignored.
        top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to None. If not None, top-k sampling is ignored.
        max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
            If not provided, it's set to the model's maximum sequence length minus 1.
        num_gen_seq (int, optional): Number of sequences to generate per prompt. Defaults to 1.
        logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
        generation_prefix (str, optional): Prefix to add before asking model to generate. Helpful to guide the generation. Defaults to "".
        compile_sampling (bool, optional): Flag indicating whether to compile the generation function. Defaults to False.
        compile_prefill (bool, optional): Flag indicating whether to compile the prefill function. Defaults to False.
        stop_tokens (Set[int], optional): Set of tokens to stop generation. Defaults to None. If not None, it will override the model's stop tokens.
        verbose (bool, optional): Flag indicating whether to print the generation throughput. Defaults to False.
    Returns:
        List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

    Note:
        This method generates assistant responses for the provided conversational dialogs.
        It employs nucleus sampling to introduce controlled randomness in text generation.
        If logprobs is True, token log probabilities are computed for each generated token.
    """
    if max_gen_len is None:
        max_gen_len = model.model.params.max_seq_len - 1
    images = None
    if isinstance(model.tokenizer.text_tokenizer, ImageTextTokenizer):
        # Vision-language model
        prompt_dicts = [
            model.tokenizer.text_tokenizer.apply_chat_template(
                dialog, generation_prefix=generation_prefix, add_generation_prompt=True
            )
            for dialog in dialogs
        ]
        prompt_tokens = [prompt_dict["input_ids"] for prompt_dict in prompt_dicts]
        num_images = sum(["pixel_values" in prompt_dict for prompt_dict in prompt_dicts])
        assert num_images in [0, len(dialogs)], "For VLM, all dialogs must either all have images or all be pure text."
        if num_images > 0:
            images = torch.cat([prompt_dict["pixel_values"] for prompt_dict in prompt_dicts], dim=0)
        else:
            images = None
    elif isinstance(model.tokenizer.text_tokenizer, TextTokenizer):
        # Text-only model
        prompt_tokens = [
            model.tokenizer.text_tokenizer.apply_chat_template(
                dialog, generation_prefix=generation_prefix, add_generation_prompt=True
            )
            for dialog in dialogs
        ]
    else:
        prompt_tokens = [model.formatter.encode_dialog_prompt(dialog) for dialog in dialogs]

    generation_tokens, generation_logprobs = model.generate(
        prompt_tokens=prompt_tokens,
        seed=seed,
        max_gen_len=max_gen_len,
        num_gen_seq=num_gen_seq,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        compile_sampling=compile_sampling,
        compile_prefill=compile_prefill,
        stop_tokens=stop_tokens,
        verbose=verbose,
        images=images,
    )

    if logprobs:
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": model.tokenizer.text_tokenizer.decode(t),
                },
                "tokens": [model.tokenizer.text_tokenizer.decode([x]) for x in t],
                "logprobs": logprobs_i,
            }
            for t, logprobs_i in zip(generation_tokens, generation_logprobs)
        ]
    return [
        {
            "generation": {
                "role": "assistant",
                "content": model.tokenizer.text_tokenizer.decode(t),
            },
        }
        for t in generation_tokens
    ]
