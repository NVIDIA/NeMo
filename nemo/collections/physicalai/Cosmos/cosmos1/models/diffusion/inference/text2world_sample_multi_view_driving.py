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

# pylint: disable=C0115,C0116,C0301

import argparse
import os

import torch

from cosmos1.models.diffusion.inference.inference_utils import add_common_arguments, remove_argument, validate_args
from cosmos1.models.diffusion.inference.world_generation_pipeline import DiffusionText2WorldMultiViewGenerationPipeline
from cosmos1.utils import log, misc
from cosmos1.utils.io import save_video

torch.enable_grad(False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)
    remove_argument(parser, "width")
    remove_argument(parser, "height")
    remove_argument(parser, "num_video_frames")
    parser.add_argument("--height", type=int, default=480, help="Height of video to sample")
    parser.add_argument("--width", type=int, default=848, help="Width of video to sample")

    parser.add_argument(
        "--num_video_frames",
        type=int,
        default=57,
        choices=[57],
        help="Number of video frames to sample, this is per-camera frame number.",
    )
    # Add text2world specific arguments
    parser.add_argument(
        "--diffusion_transformer_dir",
        type=str,
        default="Cosmos-1.0-Diffusion-7B-Text2World",
        help="DiT model weights directory name relative to checkpoint_dir",
        choices=[
            "Cosmos-1.0-Diffusion-7B-Text2World",
            "Cosmos-1.0-Diffusion-14B-Text2World",
            "Cosmos-1.1-Diffusion-7B-Text2World-Sample-Driving-Multiview",
        ],
    )
    parser.add_argument(
        "--prompt_left",
        type=str,
        default="The video is captured from a camera mounted on a car. The camera is facing to the left. ",
        help="Text prompt for generating left camera view video",
    )
    parser.add_argument(
        "--prompt_right",
        type=str,
        default="The video is captured from a camera mounted on a car. The camera is facing to the right.",
        help="Text prompt for generating right camera view video",
    )

    parser.add_argument(
        "--prompt_back",
        type=str,
        default="The video is captured from a camera mounted on a car. The camera is facing backwards.",
        help="Text prompt for generating rear camera view video",
    )
    parser.add_argument(
        "--prompt_back_left",
        type=str,
        default="The video is captured from a camera mounted on a car. The camera is facing the rear left side.",
        help="Text prompt for generating left camera view video",
    )

    parser.add_argument(
        "--prompt_back_right",
        type=str,
        default="The video is captured from a camera mounted on a car. The camera is facing the rear right side.",
        help="Text prompt for generating right camera view video",
    )
    parser.add_argument(
        "--frame_repeat_negative_condition",
        type=float,
        default=10.0,
        help="frame_repeat number to be used as negative condition",
    )

    return parser.parse_args()


def demo(cfg):
    """Run multi-view text-to-world generation demo.

    This function handles the main text-to-world generation pipeline, including:
    - Setting up the random seed for reproducibility
    - Initializing the generation pipeline with the provided configuration
    - Processing single or multiple prompts from input
    - Generating videos from text prompts
    - Saving the generated videos and corresponding prompts to disk

    Args:
        cfg (argparse.Namespace): Configuration namespace containing:
            - Model configuration (checkpoint paths, model settings)
            - Generation parameters (guidance, steps, dimensions)
            - Input/output settings (prompts, save paths)
            - Performance options (model offloading settings)

    The function will save:
        - Generated MP4 video files
        - Text files containing the processed prompts

    If guardrails block the generation, a critical log message is displayed
    and the function continues to the next prompt if available.
    """
    misc.set_random_seed(cfg.seed)
    inference_type = "text2world"
    validate_args(cfg, inference_type)

    # Initialize text2world generation model pipeline
    pipeline = DiffusionText2WorldMultiViewGenerationPipeline(
        inference_type=inference_type,
        checkpoint_dir=cfg.checkpoint_dir,
        checkpoint_name=cfg.diffusion_transformer_dir,
        offload_network=cfg.offload_diffusion_transformer,
        offload_tokenizer=cfg.offload_tokenizer,
        offload_text_encoder_model=cfg.offload_text_encoder_model,
        guidance=cfg.guidance,
        num_steps=cfg.num_steps,
        height=cfg.height,
        width=cfg.width,
        fps=cfg.fps,
        num_video_frames=cfg.num_video_frames,
        frame_repeat_negative_condition=cfg.frame_repeat_negative_condition,
        seed=cfg.seed,
    )

    prompts = {
        "prompt": cfg.prompt,
        "prompt_left": cfg.prompt_left,
        "prompt_right": cfg.prompt_right,
        "prompt_back": cfg.prompt_back,
        "prompt_back_left": cfg.prompt_back_left,
        "prompt_back_right": cfg.prompt_back_right,
    }

    os.makedirs(cfg.video_save_folder, exist_ok=True)

    # Generate video
    generated_output = pipeline.generate(prompts)
    if generated_output is None:
        log.critical("Guardrail blocked text2world generation.")
        return
    [video_grid, video], prompt = generated_output

    video_save_path = os.path.join(cfg.video_save_folder, f"{cfg.video_save_name}.mp4")
    video_grid_save_path = os.path.join(cfg.video_save_folder, f"{cfg.video_save_name}_grid.mp4")

    prompt_save_path = os.path.join(cfg.video_save_folder, f"{cfg.video_save_name}.txt")

    # Save video
    save_video(
        video=video,
        fps=cfg.fps,
        H=cfg.height,
        W=cfg.width,
        video_save_quality=10,
        video_save_path=video_save_path,
    )

    save_video(
        video=video_grid,
        fps=cfg.fps,
        H=cfg.height * 2,
        W=cfg.width * 3,
        video_save_quality=5,
        video_save_path=video_grid_save_path,
    )

    # Save prompt to text file alongside video
    with open(prompt_save_path, "wb") as f:
        for key, value in prompt.items():
            f.write(value.encode("utf-8"))
            f.write("\n".encode("utf-8"))

    log.info(f"Saved video to {video_save_path}")
    log.info(f"Saved prompt to {prompt_save_path}")


if __name__ == "__main__":
    args = parse_arguments()
    demo(args)
