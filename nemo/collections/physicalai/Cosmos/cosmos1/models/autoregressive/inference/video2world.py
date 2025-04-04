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

import argparse
import os

import imageio
import torch

from cosmos1.models.autoregressive.inference.world_generation_pipeline import ARVideo2WorldGenerationPipeline
from cosmos1.models.autoregressive.utils.inference import add_common_arguments, load_vision_input, validate_args
from cosmos1.utils import log
from cosmos1.utils.io import read_prompts_from_file


def parse_args():
    parser = argparse.ArgumentParser(description="Prompted video to world generation demo script")
    add_common_arguments(parser)
    parser.add_argument(
        "--ar_model_dir",
        type=str,
        default="Cosmos-1.0-Autoregressive-5B-Video2World",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default="text_and_video",
        choices=["text_and_image", "text_and_video"],
        help="Input types",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for generating a single video",
    )
    parser.add_argument(
        "--offload_text_encoder_model",
        action="store_true",
        help="Offload T5 model after inference",
    )
    args = parser.parse_args()
    return args


def main(args):
    """Run prompted video-to-world generation demo.

    This function handles the main video-to-world generation pipeline, including:
    - Setting up the random seed for reproducibility
    - Initializing the generation pipeline with the provided configuration
    - Processing single or multiple prompts/images/videos from input
    - Generating videos from prompts and images/videos
    - Saving the generated videos and corresponding prompts to disk

    Args:
        cfg (argparse.Namespace): Configuration namespace containing:
            - Model configuration (checkpoint paths, model settings)
            - Generation parameters (temperature, top_p)
            - Input/output settings (images/videos, save paths)
            - Performance options (model offloading settings)

    The function will save:
        - Generated MP4 video files

    If guardrails block the generation, a critical log message is displayed
    and the function continues to the next prompt if available.
    """
    inference_type = "video2world"  # When the inference_type is "video2world", AR model takes both text and video as input, the world generation is based on the input text prompt and video
    sampling_config = validate_args(args, inference_type)

    # Initialize prompted base generation model pipeline
    pipeline = ARVideo2WorldGenerationPipeline(
        inference_type=inference_type,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.ar_model_dir,
        disable_diffusion_decoder=args.disable_diffusion_decoder,
        offload_guardrail_models=args.offload_guardrail_models,
        offload_diffusion_decoder=args.offload_diffusion_decoder,
        offload_network=args.offload_ar_model,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder_model=args.offload_text_encoder_model,
    )

    # Load input image(s) or video(s)
    input_videos = load_vision_input(
        input_type=args.input_type,
        batch_input_path=args.batch_input_path,
        input_image_or_video_path=args.input_image_or_video_path,
        data_resolution=args.data_resolution,
        num_input_frames=args.num_input_frames,
    )
    # Load input prompt(s)
    if args.batch_input_path:
        prompts_list = read_prompts_from_file(args.batch_input_path)
    else:
        prompts_list = [{"visual_input": args.input_image_or_video_path, "prompt": args.prompt}]

    # Iterate through prompts
    for idx, prompt_entry in enumerate(prompts_list):
        video_path = prompt_entry["visual_input"]
        input_filename = os.path.basename(video_path)

        # Check if video exists in loaded videos
        if input_filename not in input_videos:
            log.critical(f"Input file {input_filename} not found, skipping prompt.")
            continue

        inp_vid = input_videos[input_filename]
        inp_prompt = prompt_entry["prompt"]

        # Generate video
        log.info(f"Run with input: {prompt_entry}")
        out_vid = pipeline.generate(
            inp_prompt=inp_prompt,
            inp_vid=inp_vid,
            num_input_frames=args.num_input_frames,
            seed=args.seed,
            sampling_config=sampling_config,
        )
        if out_vid is None:
            log.critical("Guardrail blocked video2world generation.")
            continue

        # Save video
        if args.input_image_or_video_path:
            out_vid_path = os.path.join(args.video_save_folder, f"{args.video_save_name}.mp4")
        else:
            out_vid_path = os.path.join(args.video_save_folder, f"{idx}.mp4")
        imageio.mimsave(out_vid_path, out_vid, fps=25)

        log.info(f"Saved video to {out_vid_path}")


if __name__ == "__main__":
    torch._C._jit_set_texpr_fuser_enabled(False)
    args = parse_args()
    main(args)
