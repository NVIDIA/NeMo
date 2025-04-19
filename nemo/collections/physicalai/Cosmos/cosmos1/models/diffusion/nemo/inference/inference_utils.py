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

# pylint: disable=C0115,C0116,C0301

import os

import imageio
import numpy as np
import torch

from cosmos1.models.autoregressive.model import AutoRegressiveModel
from cosmos1.models.diffusion.config.ctrl.augmentors import BilateralOnlyBlurAugmentorConfig
from cosmos1.models.diffusion.datasets.augmentors.control_input import get_augmentor_for_eval
from cosmos1.models.diffusion.inference.inference_utils import read_video_or_image_into_frames_BCTHW
from cosmos1.models.diffusion.prompt_upsampler.text2world_prompt_upsampler_inference import (
    create_prompt_upsampler,
    run_chat_completion,
)
from cosmos1.models.guardrail.common.presets import (
    create_text_guardrail_runner,
    create_video_guardrail_runner,
    run_text_guardrail,
    run_video_guardrail,
)
from cosmos1.utils import log


def get_upsampled_prompt(
    prompt_upsampler_model: AutoRegressiveModel, input_prompt: str, temperature: float = 0.01
) -> str:
    """
    Get upsampled prompt from the prompt upsampler model instance.

    Args:
        prompt_upsampler_model: The prompt upsampler model instance.
        input_prompt (str): Original prompt to upsample.
        temperature (float): Temperature for generation (default: 0.01).

    Returns:
        str: The upsampled prompt.
    """
    dialogs = [
        [
            {
                "role": "user",
                "content": f"Upsample the short caption to a long caption: {input_prompt}",
            }
        ]
    ]

    upsampled_prompt = run_chat_completion(prompt_upsampler_model, dialogs, temperature=temperature)
    return upsampled_prompt


def print_rank_0(string: str):
    rank = torch.distributed.get_rank()
    if rank == 0:
        log.info(string)


def process_prompt(
    prompt: str,
    checkpoint_dir: str,
    prompt_upsampler_dir: str,
    guardrails_dir: str,
    image_path: str = None,
    enable_prompt_upsampler: bool = True,
) -> str:
    """
    Handle prompt upsampling if enabled, then run guardrails to ensure safety.

    Args:
        prompt (str): The original text prompt.
        checkpoint_dir (str): Base checkpoint directory.
        prompt_upsampler_dir (str): Directory containing prompt upsampler weights.
        guardrails_dir (str): Directory containing guardrails weights.
        image_path (str, optional): Path to an image, if any (not implemented for upsampling).
        enable_prompt_upsampler (bool): Whether to enable prompt upsampling.

    Returns:
        str: The upsampled prompt or original prompt if upsampling is disabled or fails.
    """

    text_guardrail = create_text_guardrail_runner(os.path.join(checkpoint_dir, guardrails_dir))

    # Check if the prompt is safe
    is_safe = run_text_guardrail(str(prompt), text_guardrail)
    if not is_safe:
        raise ValueError("Guardrail blocked world generation.")

    if enable_prompt_upsampler:
        if image_path:
            raise NotImplementedError("Prompt upsampling is not supported for image generation")
        else:
            prompt_upsampler = create_prompt_upsampler(
                checkpoint_dir=os.path.join(checkpoint_dir, prompt_upsampler_dir)
            )
            upsampled_prompt = get_upsampled_prompt(prompt_upsampler, prompt)
            print_rank_0(f"Original prompt: {prompt}\nUpsampled prompt: {upsampled_prompt}\n")
            del prompt_upsampler

            # Re-check the upsampled prompt
            is_safe = run_text_guardrail(str(upsampled_prompt), text_guardrail)
            if not is_safe:
                raise ValueError("Guardrail blocked world generation.")

            return upsampled_prompt
    else:
        return prompt


def save_video(
    grid: np.ndarray,
    fps: int,
    H: int,
    W: int,
    video_save_quality: int,
    video_save_path: str,
    checkpoint_dir: str,
    guardrails_dir: str,
):
    """
    Save video frames to file, applying a safety check before writing.

    Args:
        grid (np.ndarray): Video frames array [T, H, W, C].
        fps (int): Frames per second.
        H (int): Frame height.
        W (int): Frame width.
        video_save_quality (int): Video encoding quality (0-10).
        video_save_path (str): Output video file path.
        checkpoint_dir (str): Directory containing model checkpoints.
        guardrails_dir (str): Directory containing guardrails weights.
    """
    video_classifier_guardrail = create_video_guardrail_runner(os.path.join(checkpoint_dir, guardrails_dir))

    # Safety check on the entire video
    grid = run_video_guardrail(grid, video_classifier_guardrail)

    kwargs = {
        "fps": fps,
        "quality": video_save_quality,
        "macro_block_size": 1,
        "ffmpeg_params": ["-s", f"{W}x{H}"],
        "output_params": ["-f", "mp4"],
    }

    imageio.mimsave(video_save_path, grid, "mp4", **kwargs)


def get_ctrl_batch_nemo(
    model,
    data_batch,
    num_video_frames,
    input_image_or_video_path,
    control_weight,
    blur_str,
    no_preserve_color,
    state_shape,
    spatial_compression_factor,
    hint_key="control_input_canny",
    tokenizer=None,
):
    """Prepare complete input batch for video generation including latent dimensions.

    Args:
        model: Diffusion model instance

    Returns:
        - data_batch (dict): Complete model input batch
    """

    H, W = (
        state_shape[-2] * spatial_compression_factor,
        state_shape[-1] * spatial_compression_factor,
    )

    input_path_format = input_image_or_video_path.split(".")[-1]
    input_frames = read_video_or_image_into_frames_BCTHW(
        input_image_or_video_path,
        input_path_format=input_path_format,
        H=H,
        W=W,
    )[:, :, :num_video_frames]
    T = input_frames.shape[2]
    if T < num_video_frames:
        pad_frames = input_frames[:, :, -1:].repeat(1, 1, num_video_frames - T, 1, 1)
        input_frames = torch.cat([input_frames, pad_frames], dim=2)

    add_control_input = get_augmentor_for_eval(
        input_key="video",
        output_key=hint_key,
        preset_strength=blur_str,
        blur_config=BilateralOnlyBlurAugmentorConfig[blur_str],
    )
    if no_preserve_color:
        model.config.hint_mask = [True, False]
    else:
        model.config.hint_mask = [False, True]

    data_batch["hint_key"] = hint_key
    data_batch["video"] = ((input_frames.cpu().float().numpy()[0] + 1) / 2 * 255).astype(np.uint8)
    control_input = add_control_input(data_batch)[hint_key]

    data_batch["video"] = input_frames
    data_batch[hint_key] = control_input[None].bfloat16().cuda() / 255 * 2 - 1
    if tokenizer is not None:
        data_batch["latent_hint"] = model.encode_latent(data_batch, tokenizer)
    else:
        data_batch["latent_hint"] = model.encode_latent(data_batch)
    data_batch["control_weight"] = control_weight

    return data_batch
