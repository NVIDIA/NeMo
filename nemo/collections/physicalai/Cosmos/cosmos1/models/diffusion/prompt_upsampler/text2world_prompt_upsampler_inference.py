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

"""
This demo script is used to run inference for Cosmos-1.0-Prompt-Upsampler-12B-Text2World.
Command:
    PYTHONPATH=$(pwd) python cosmos1/models/diffusion/prompt_upsampler/text2world_prompt_upsampler_inference.py

"""
import argparse
import os
import re

from cosmos1.models.autoregressive.configs.base.model_config import create_text_model_config
from cosmos1.models.autoregressive.model import AutoRegressiveModel
from cosmos1.models.diffusion.prompt_upsampler.inference import chat_completion
from cosmos1.models.guardrail.common import presets as guardrail_presets
from cosmos1.utils import log


def create_prompt_upsampler(checkpoint_dir: str) -> AutoRegressiveModel:
    model_config, tokenizer_config = create_text_model_config(
        model_ckpt_path=os.path.join(checkpoint_dir, "model.pt"),
        tokenizer_path=os.path.join(checkpoint_dir),
        model_family="mistral",
        model_size="12b",
        is_instruct_model=True,
        max_batch_size=1,
        rope_dim="1D",
        add_special_tokens=True,
        max_seq_len=1024,
        pytorch_rope_version="v1",
    )
    log.debug(f"Text prompt upsampler model config: {model_config}")

    # Create and return a LLM instance
    return AutoRegressiveModel.build(
        model_config=model_config,
        tokenizer_config=tokenizer_config,
    ).to("cuda")


def run_chat_completion(model: AutoRegressiveModel, input: str, temperature: float = 0.01):
    """
    text2world prompt upsampler model is finetuned for chat.
    During training, the context window for the initial prompt upsampler models is 512 tokens. For inference, we set max_seq_len to 1024 to accommodate longer inputs.
    Setting `max_gen_len` is optional as the finetuned models can naturally determine when to stop generating.
    """

    dialogs = [[{"role": "user", "content": f"Upsample the short caption to a long caption: {str(input)}"}]]

    results = chat_completion(
        model,
        dialogs,
        max_gen_len=512,
        temperature=temperature,
        top_p=None,
        top_k=None,
        logprobs=False,
    )
    upsampled_prompt = str(clean_text(results[0]["generation"]["content"]))
    return upsampled_prompt


def clean_text(text: str) -> str:
    """Clean the text by removing prefixes, suffixes, formatting markers, and normalizing whitespace."""
    # Replace all variations of newlines with a space
    text = text.replace("\n", " ").replace("\r", " ")

    # Use a regex to find sections of the form '- **...**'
    pattern = r"(- \*\*)(.*?)(\*\*)"

    def replacement(match: re.Match[str]) -> str:
        content = match.group(2)  # The text inside - ** and **
        words = re.findall(r"\w+", content)
        if len(words) < 10:
            # If fewer than 10 words, remove the entire '- **...**' portion
            return ""
        else:
            # If 10 or more words, keep the entire section as it is
            return match.group(0)

    text = re.sub(pattern, replacement, text)

    # Remove common prefixes
    prefixes = ["Caption:", "#####", "####", "- ", "* ", ","]
    for prefix in prefixes:
        # lstrip(prefix) won't strip entire strings, but character sets.
        # For more reliable prefix removal, do:
        if text.startswith(prefix):
            text = text[len(prefix) :].lstrip()

    # Remove extra spaces
    text = " ".join(text.split())

    # Strip any remaining leading/trailing punctuation, whitespace, and quotes
    text = text.strip(' -,*:"\'"“”')

    return text


def parse_args():
    parser = argparse.ArgumentParser(description="Run prompt upsampler inference")
    parser.add_argument("--input", type=str, default="A dog is playing with a ball.")
    parser.add_argument("--temperature", type=float, default=0.01, help="Inference temperature")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Base directory containing model checkpoints"
    )
    parser.add_argument(
        "--prompt_upsampler_dir",
        type=str,
        default="Cosmos-1.0-Prompt-Upsampler-12B-Text2World",
        help="Prompt upsampler weights directory relative to checkpoint_dir",
    )
    parser.add_argument(
        "--guardrail_dir",
        type=str,
        default="Cosmos-1.0-Guardrail",
        help="Guardrail weights directory relative to checkpoint_dir",
    )
    return parser.parse_args()


def main(args):
    guardrail_runner = guardrail_presets.create_text_guardrail_runner(
        os.path.join(args.checkpoint_dir, args.guardrail_dir)
    )
    is_safe = guardrail_presets.run_text_guardrail(args.input, guardrail_runner)
    if not is_safe:
        log.critical("Input text prompt is not safe.")
        return

    prompt_upsampler = create_prompt_upsampler(os.path.join(args.checkpoint_dir, args.prompt_upsampler_dir))
    upsampled_prompt = run_chat_completion(prompt_upsampler, args.input, temperature=args.temperature)
    is_safe = guardrail_presets.run_text_guardrail(upsampled_prompt, guardrail_runner)
    if not is_safe:
        log.critical("Upsampled text prompt is not safe.")
        return

    log.info(f"Upsampled prompt: {upsampled_prompt}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
