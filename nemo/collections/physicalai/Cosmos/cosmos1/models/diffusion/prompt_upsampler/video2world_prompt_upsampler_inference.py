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
This demo script is used to run inference for Pixtral-12B.
Command:
    PYTHONPATH=$(pwd) python cosmos1/models/diffusion/prompt_upsampler/video2world_prompt_upsampler_inference.py

"""

import argparse
import os
from math import ceil

from PIL import Image

from cosmos1.models.autoregressive.configs.base.model_config import create_vision_language_model_config
from cosmos1.models.autoregressive.model import AutoRegressiveModel
from cosmos1.models.diffusion.prompt_upsampler.inference import chat_completion
from cosmos1.models.guardrail.common import presets as guardrail_presets
from cosmos1.utils import log
from cosmos1.utils.io import load_from_fileobj


def create_vlm_prompt_upsampler(
    checkpoint_dir: str, tokenizer_ckpt_path: str = "mistral-community/pixtral-12b"
) -> AutoRegressiveModel:
    """
    Load the fine-tuned pixtral model for SimReady.
    If pixtral_ckpt is not provided, use the pretrained checkpoint.
    """
    model_ckpt_path = os.path.join(checkpoint_dir, "model.pt")
    model_config, tokenizer_config = create_vision_language_model_config(
        model_ckpt_path=model_ckpt_path,
        tokenizer_ckpt_path=tokenizer_ckpt_path,
        model_family="pixtral",
        model_size="12b",
        is_instruct_model=True,
        max_batch_size=1,
        max_seq_len=4300,
        pytorch_rope_version="v1",
    )
    # during instantiate, the weights will be downloaded (if not already cached) and loaded
    return AutoRegressiveModel.build(
        model_config=model_config,
        tokenizer_config=tokenizer_config,
    ).to("cuda")


def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """
    Ensure that the image is no larger than max_size in both dimensions.
    """
    image_width, image_height = image.size
    max_width, max_height = max_size, max_size
    ratio = max(image_width / max_width, image_height / max_height)
    if ratio > 1:
        image = image.resize((ceil(image_width / ratio), ceil(image_height / ratio)))
    return image


def prepare_dialog(image_or_video_path: str) -> list[dict]:
    if image_or_video_path.endswith(".mp4"):
        video_np, _ = load_from_fileobj(image_or_video_path, format="mp4")
        image_frame = video_np[-1]
        image = Image.fromarray(image_frame)
    else:
        image: Image.Image = Image.open(image_or_video_path)

    image = resize_image(image, max_size=1024)
    prompt = """\
Your task is to transform a given prompt into a refined and concise video description, no more than 150 words.
Focus only on the content, no filler words or descriptions on the style. Never mention things outside the video.
    """.strip()

    return [
        {
            "role": "user",
            "content": "[IMG]\n" + prompt,
            "images": [image],
        }
    ]


def run_chat_completion(pixtral: AutoRegressiveModel, dialog: list[dict], **inference_args) -> str:
    default_args = {
        "max_gen_len": 400,
        "temperature": 0,
        "top_p": 0.9,
        "logprobs": False,
        "compile_sampling": False,
        "compile_prefill": False,
    }
    default_args.update(inference_args)
    results = chat_completion(
        pixtral,
        [dialog],
        **default_args,
    )
    assert len(results) == 1
    upsampled_prompt = str(results[0]["generation"]["content"])
    return upsampled_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Run prompt upsampler inference")
    parser.add_argument(
        "--image_or_video_path", type=str, default="cosmos1/models/diffusion/assets/v1p0/video2world_input0.jpg"
    )
    parser.add_argument("--temperature", type=float, default=0.01, help="Inference temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p value for top-p sampling")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Base directory containing model checkpoints"
    )
    parser.add_argument(
        "--prompt_upsampler_dir",
        type=str,
        default="Pixtral-12B",
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

    pixtral = create_vlm_prompt_upsampler(os.path.join(args.checkpoint_dir, args.prompt_upsampler_dir))
    dialog = prepare_dialog(args.image_or_video_path)
    upsampled_prompt = run_chat_completion(
        pixtral,
        dialog,
        max_gen_len=400,
        temperature=args.temperature,
        top_p=args.top_p,
        logprobs=False,
    )
    is_safe = guardrail_presets.run_text_guardrail(upsampled_prompt, guardrail_runner)
    if not is_safe:
        log.critical("Upsampled text prompt is not safe.")
        return

    log.info(f"Upsampled prompt: {upsampled_prompt}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
