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

import argparse
import importlib
from contextlib import contextmanager
from typing import List, NamedTuple, Optional, Tuple

import einops
import imageio
import numpy as np
import torch
import torchvision.transforms.functional as transforms_F

from cosmos1.models.diffusion.model.model_sample_multiview_driving import DiffusionMultiCameraV2WModel
from cosmos1.models.diffusion.model.model_t2w import DiffusionT2WModel
from cosmos1.models.diffusion.model.model_v2w import DiffusionV2WModel
from cosmos1.utils import log, misc
from cosmos1.utils.config_helper import get_config_module, override
from cosmos1.utils.io import load_from_fileobj

TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])
if TORCH_VERSION >= (1, 11):
    from torch.ao import quantization
    from torch.ao.quantization import FakeQuantizeBase, ObserverBase
elif (
    TORCH_VERSION >= (1, 8)
    and hasattr(torch.quantization, "FakeQuantizeBase")
    and hasattr(torch.quantization, "ObserverBase")
):
    from torch import quantization
    from torch.quantization import FakeQuantizeBase, ObserverBase

DEFAULT_AUGMENT_SIGMA = 0.001


def add_common_arguments(parser):
    """Add common command line arguments for text2world and video2world generation.

    Args:
        parser (ArgumentParser): Argument parser to add arguments to

    The arguments include:
    - checkpoint_dir: Base directory containing model weights
    - tokenizer_dir: Directory containing tokenizer weights
    - video_save_name: Output video filename for single video generation
    - video_save_folder: Output directory for batch video generation
    - prompt: Text prompt for single video generation
    - batch_input_path: Path to JSONL file with input prompts for batch video generation
    - negative_prompt: Text prompt describing undesired attributes
    - num_steps: Number of diffusion sampling steps
    - guidance: Classifier-free guidance scale
    - num_video_frames: Number of frames to generate
    - height/width: Output video dimensions
    - fps: Output video frame rate
    - seed: Random seed for reproducibility
    - Various model offloading flags
    """
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Base directory containing model checkpoints"
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="Cosmos-1.0-Tokenizer-CV8x8x8",
        help="Tokenizer weights directory relative to checkpoint_dir",
    )
    parser.add_argument(
        "--video_save_name",
        type=str,
        default="output",
        help="Output filename for generating a single video",
    )
    parser.add_argument(
        "--video_save_folder",
        type=str,
        default="outputs/",
        help="Output folder for generating a batch of videos",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for generating a single video",
    )
    parser.add_argument(
        "--batch_input_path",
        type=str,
        help="Path to a JSONL file of input prompts for generating a batch of videos",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
        "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
        "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
        "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special "
        "effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and "
        "flickering. Overall, the video is of poor quality.",
        help="Negative prompt for the video",
    )
    parser.add_argument("--num_steps", type=int, default=35, help="Number of diffusion sampling steps")
    parser.add_argument("--guidance", type=float, default=7, help="Guidance scale value")
    parser.add_argument(
        "--num_video_frames", type=int, default=121, choices=[121], help="Number of video frames to sample"
    )
    parser.add_argument("--height", type=int, default=704, help="Height of video to sample")
    parser.add_argument("--width", type=int, default=1280, help="Width of video to sample")
    parser.add_argument("--fps", type=int, default=24, help="FPS of the sampled video")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--disable_prompt_upsampler",
        action="store_true",
        help="Disable prompt upsampling",
    )
    parser.add_argument(
        "--offload_diffusion_transformer",
        action="store_true",
        help="Offload DiT after inference",
    )
    parser.add_argument(
        "--offload_tokenizer",
        action="store_true",
        help="Offload tokenizer after inference",
    )
    parser.add_argument(
        "--offload_text_encoder_model",
        action="store_true",
        help="Offload text encoder model after inference",
    )
    parser.add_argument(
        "--offload_prompt_upsampler",
        action="store_true",
        help="Offload prompt upsampler after inference",
    )
    parser.add_argument(
        "--offload_guardrail_models",
        action="store_true",
        help="Offload guardrail models after inference",
    )


# Function to fully remove an argument
def remove_argument(parser, arg_name):
    # Get a list of actions to remove
    actions_to_remove = [action for action in parser._actions if action.dest == arg_name]

    for action in actions_to_remove:
        # Remove action from parser._actions
        parser._actions.remove(action)

        # Remove option strings
        for option_string in action.option_strings:
            parser._option_string_actions.pop(option_string, None)


def validate_args(args: argparse.Namespace, inference_type: str) -> None:
    """Validate command line arguments for text2world and video2world generation."""
    assert inference_type in [
        "text2world",
        "video2world",
    ], "Invalid inference_type, must be 'text2world' or 'video2world'"

    # Validate prompt/image/video args for single or batch generation
    if inference_type == "text2world" or (inference_type == "video2world" and args.disable_prompt_upsampler):
        assert args.prompt or args.batch_input_path, "--prompt or --batch_input_path must be provided."
    if inference_type == "video2world" and not args.batch_input_path:
        assert (
            args.input_image_or_video_path
        ), "--input_image_or_video_path must be provided for single video generation."


class _IncompatibleKeys(
    NamedTuple(
        "IncompatibleKeys",
        [
            ("missing_keys", List[str]),
            ("unexpected_keys", List[str]),
            ("incorrect_shapes", List[Tuple[str, Tuple[int], Tuple[int]]]),
        ],
    )
):
    pass


def non_strict_load_model(model: torch.nn.Module, checkpoint_state_dict: dict) -> _IncompatibleKeys:
    """Load a model checkpoint with non-strict matching, handling shape mismatches.

    Args:
        model (torch.nn.Module): Model to load weights into
        checkpoint_state_dict (dict): State dict from checkpoint

    Returns:
        _IncompatibleKeys: Named tuple containing:
            - missing_keys: Keys present in model but missing from checkpoint
            - unexpected_keys: Keys present in checkpoint but not in model
            - incorrect_shapes: Keys with mismatched tensor shapes

    The function handles special cases like:
    - Uninitialized parameters
    - Quantization observers
    - TransformerEngine FP8 states
    """
    # workaround https://github.com/pytorch/pytorch/issues/24139
    model_state_dict = model.state_dict()
    incorrect_shapes = []
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            if "_extra_state" in k:  # Key introduced by TransformerEngine for FP8
                log.debug(f"Skipping key {k} introduced by TransformerEngine for FP8 in the checkpoint.")
                continue
            model_param = model_state_dict[k]
            # Allow mismatch for uninitialized parameters
            if TORCH_VERSION >= (1, 8) and isinstance(model_param, torch.nn.parameter.UninitializedParameter):
                continue
            if not isinstance(model_param, torch.Tensor):
                raise ValueError(
                    f"Find non-tensor parameter {k} in the model. type: {type(model_param)} {type(checkpoint_state_dict[k])}, please check if this key is safe to skip or not."
                )

            shape_model = tuple(model_param.shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                has_observer_base_classes = (
                    TORCH_VERSION >= (1, 8)
                    and hasattr(quantization, "ObserverBase")
                    and hasattr(quantization, "FakeQuantizeBase")
                )
                if has_observer_base_classes:
                    # Handle the special case of quantization per channel observers,
                    # where buffer shape mismatches are expected.
                    def _get_module_for_key(model: torch.nn.Module, key: str) -> torch.nn.Module:
                        # foo.bar.param_or_buffer_name -> [foo, bar]
                        key_parts = key.split(".")[:-1]
                        cur_module = model
                        for key_part in key_parts:
                            cur_module = getattr(cur_module, key_part)
                        return cur_module

                    cls_to_skip = (
                        ObserverBase,
                        FakeQuantizeBase,
                    )
                    target_module = _get_module_for_key(model, k)
                    if isinstance(target_module, cls_to_skip):
                        # Do not remove modules with expected shape mismatches
                        # them from the state_dict loading. They have special logic
                        # in _load_from_state_dict to handle the mismatches.
                        continue

                incorrect_shapes.append((k, shape_checkpoint, shape_model))
                checkpoint_state_dict.pop(k)
    incompatible = model.load_state_dict(checkpoint_state_dict, strict=False)
    # Remove keys with "_extra_state" suffix, which are non-parameter items introduced by TransformerEngine for FP8 handling
    missing_keys = [k for k in incompatible.missing_keys if "_extra_state" not in k]
    unexpected_keys = [k for k in incompatible.unexpected_keys if "_extra_state" not in k]
    return _IncompatibleKeys(
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        incorrect_shapes=incorrect_shapes,
    )


@contextmanager
def skip_init_linear():
    # skip init of nn.Linear
    orig_reset_parameters = torch.nn.Linear.reset_parameters
    torch.nn.Linear.reset_parameters = lambda x: x
    xavier_uniform_ = torch.nn.init.xavier_uniform_
    torch.nn.init.xavier_uniform_ = lambda x: x
    yield
    torch.nn.Linear.reset_parameters = orig_reset_parameters
    torch.nn.init.xavier_uniform_ = xavier_uniform_


def load_model_by_config(
    config_job_name,
    config_file="projects/cosmos_video/config/config.py",
    model_class=DiffusionT2WModel,
):
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()

    config = override(config, ["--", f"experiment={config_job_name}"])

    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore

    # Initialize model
    with skip_init_linear():
        model = model_class(config.model)
    return model


def load_network_model(model: DiffusionT2WModel, ckpt_path: str):
    with skip_init_linear():
        model.set_up_model()
    net_state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    log.debug(non_strict_load_model(model.model, net_state_dict))
    model.cuda()


def load_tokenizer_model(model: DiffusionT2WModel, tokenizer_dir: str):
    with skip_init_linear():
        model.set_up_tokenizer(tokenizer_dir)
    model.cuda()


def prepare_data_batch(
    height: int,
    width: int,
    num_frames: int,
    fps: int,
    prompt_embedding: torch.Tensor,
    negative_prompt_embedding: Optional[torch.Tensor] = None,
):
    """Prepare input batch tensors for video generation.

    Args:
        height (int): Height of video frames
        width (int): Width of video frames
        num_frames (int): Number of frames to generate
        fps (int): Frames per second
        prompt_embedding (torch.Tensor): Encoded text prompt embeddings
        negative_prompt_embedding (torch.Tensor, optional): Encoded negative prompt embeddings

    Returns:
        dict: Batch dictionary containing:
            - video: Zero tensor of target video shape
            - t5_text_mask: Attention mask for text embeddings
            - image_size: Target frame dimensions
            - fps: Target frame rate
            - num_frames: Number of frames
            - padding_mask: Frame padding mask
            - t5_text_embeddings: Prompt embeddings
            - neg_t5_text_embeddings: Negative prompt embeddings (if provided)
            - neg_t5_text_mask: Mask for negative embeddings (if provided)
    """
    # Create base data batch
    data_batch = {
        "video": torch.zeros((1, 3, num_frames, height, width), dtype=torch.uint8).cuda(),
        "t5_text_mask": torch.ones(1, 512, dtype=torch.bfloat16).cuda(),
        "image_size": torch.tensor([[height, width, height, width]] * 1, dtype=torch.bfloat16).cuda(),
        "fps": torch.tensor([fps] * 1, dtype=torch.bfloat16).cuda(),
        "num_frames": torch.tensor([num_frames] * 1, dtype=torch.bfloat16).cuda(),
        "padding_mask": torch.zeros((1, 1, height, width), dtype=torch.bfloat16).cuda(),
    }

    # Handle text embeddings

    t5_embed = prompt_embedding.to(dtype=torch.bfloat16).cuda()
    data_batch["t5_text_embeddings"] = t5_embed

    if negative_prompt_embedding is not None:
        neg_t5_embed = negative_prompt_embedding.to(dtype=torch.bfloat16).cuda()
        data_batch["neg_t5_text_embeddings"] = neg_t5_embed
        data_batch["neg_t5_text_mask"] = torch.ones(1, 512, dtype=torch.bfloat16).cuda()

    return data_batch


def get_video_batch(model, prompt_embedding, negative_prompt_embedding, height, width, fps, num_video_frames):
    """Prepare complete input batch for video generation including latent dimensions.

    Args:
        model: Diffusion model instance
        prompt_embedding (torch.Tensor): Text prompt embeddings
        negative_prompt_embedding (torch.Tensor): Negative prompt embeddings
        height (int): Output video height
        width (int): Output video width
        fps (int): Output video frame rate
        num_video_frames (int): Number of frames to generate

    Returns:
        tuple:
            - data_batch (dict): Complete model input batch
            - state_shape (list): Shape of latent state [C,T,H,W] accounting for VAE compression
    """
    raw_video_batch = prepare_data_batch(
        height=height,
        width=width,
        num_frames=num_video_frames,
        fps=fps,
        prompt_embedding=prompt_embedding,
        negative_prompt_embedding=negative_prompt_embedding,
    )
    state_shape = [
        model.tokenizer.channel,
        model.tokenizer.get_latent_num_frames(num_video_frames),
        height // model.tokenizer.spatial_compression_factor,
        width // model.tokenizer.spatial_compression_factor,
    ]
    return raw_video_batch, state_shape


def get_video_batch_for_multi_camera_model(
    model, prompt_embedding, height, width, fps, num_video_frames, frame_repeat_negative_condition
):
    """Prepare complete input batch for video generation including latent dimensions.

    Args:
        model: Diffusion model instance
        prompt_embedding (torch.Tensor): Text prompt embeddings
        height (int): Output video height
        width (int): Output video width
        fps (int): Output video frame rate
        num_video_frames (int): Number of frames to generate
        frame_repeat_negative_condition (int): Number of frames to generate

    Returns:
        tuple:
            - data_batch (dict): Complete model input batch
            - state_shape (list): Shape of latent state [C,T,H,W] accounting for VAE compression
    """
    n_cameras = len(prompt_embedding)
    prompt_embedding = einops.rearrange(torch.cat(prompt_embedding), "n t d -> (n t) d").unsqueeze(0)
    raw_video_batch = prepare_data_batch(
        height=height,
        width=width,
        num_frames=num_video_frames,
        fps=fps,
        prompt_embedding=prompt_embedding,
    )
    if frame_repeat_negative_condition != -1:
        frame_repeat = torch.zeros(n_cameras)
        frame_repeat[-1] = frame_repeat_negative_condition
        frame_repeat[-2] = frame_repeat_negative_condition
        raw_video_batch["frame_repeat"] = frame_repeat.unsqueeze(0).to(dtype=torch.bfloat16).cuda()
    state_shape = [
        model.tokenizer.channel,
        model.tokenizer.get_latent_num_frames(int(num_video_frames / n_cameras)) * n_cameras,
        height // model.tokenizer.spatial_compression_factor,
        width // model.tokenizer.spatial_compression_factor,
    ]
    return raw_video_batch, state_shape


def generate_world_from_text(
    model: DiffusionT2WModel,
    state_shape: list[int],
    is_negative_prompt: bool,
    data_batch: dict,
    guidance: float,
    num_steps: int,
    seed: int,
):
    """Generate video from text prompt using diffusion model.

    Args:
        model (DiffusionT2WModel): Text-to-video diffusion model
        state_shape (list[int]): Latent state dimensions [C,T,H,W]
        is_negative_prompt (bool): Whether negative prompt is provided
        data_batch (dict): Model input batch with embeddings
        guidance (float): Classifier-free guidance scale
        num_steps (int): Number of diffusion sampling steps
        seed (int): Random seed for reproducibility

    Returns:
        np.ndarray: Generated video frames [T,H,W,C], range [0,255]

    The function:
    1. Initializes random latent with maximum noise
    2. Performs guided diffusion sampling
    3. Decodes latents to pixel space
    """
    x_sigma_max = (
        misc.arch_invariant_rand(
            (1,) + tuple(state_shape),
            torch.float32,
            model.tensor_kwargs["device"],
            seed,
        )
        * model.sde.sigma_max
    )

    # Generate video
    sample = model.generate_samples_from_batch(
        data_batch,
        guidance=guidance,
        state_shape=state_shape,
        num_steps=num_steps,
        is_negative_prompt=is_negative_prompt,
        seed=seed,
        x_sigma_max=x_sigma_max,
    )

    return sample


def generate_world_from_video(
    model: DiffusionV2WModel,
    state_shape: list[int],
    is_negative_prompt: bool,
    data_batch: dict,
    guidance: float,
    num_steps: int,
    seed: int,
    condition_latent: torch.Tensor,
    num_input_frames: int,
) -> Tuple[np.array, list, list]:
    """Generate video using a conditioning video/image input.

    Args:
        model (DiffusionV2WModel): The diffusion model instance
        state_shape (list[int]): Shape of the latent state [C,T,H,W]
        is_negative_prompt (bool): Whether negative prompt is provided
        data_batch (dict): Batch containing model inputs including text embeddings
        guidance (float): Classifier-free guidance scale for sampling
        num_steps (int): Number of diffusion sampling steps
        seed (int): Random seed for generation
        condition_latent (torch.Tensor): Latent tensor from conditioning video/image file
        num_input_frames (int): Number of input frames

    Returns:
        np.array: Generated video frames in shape [T,H,W,C], range [0,255]
    """
    assert not model.config.conditioner.video_cond_bool.sample_tokens_start_from_p_or_i, "not supported"
    augment_sigma = DEFAULT_AUGMENT_SIGMA

    if condition_latent.shape[2] < state_shape[1]:
        # Padding condition latent to state shape
        b, c, t, h, w = condition_latent.shape
        condition_latent = torch.cat(
            [
                condition_latent,
                condition_latent.new_zeros(b, c, state_shape[1] - t, h, w),
            ],
            dim=2,
        ).contiguous()
    num_of_latent_condition = compute_num_latent_frames(model, num_input_frames)
    x_sigma_max = (
        misc.arch_invariant_rand(
            (1,) + tuple(state_shape),
            torch.float32,
            model.tensor_kwargs["device"],
            seed,
        )
        * model.sde.sigma_max
    )

    sample = model.generate_samples_from_batch(
        data_batch,
        guidance=guidance,
        state_shape=state_shape,
        num_steps=num_steps,
        is_negative_prompt=is_negative_prompt,
        seed=seed,
        condition_latent=condition_latent,
        num_condition_t=num_of_latent_condition,
        condition_video_augment_sigma_in_inference=augment_sigma,
        x_sigma_max=x_sigma_max,
    )
    return sample


def read_video_or_image_into_frames_BCTHW(
    input_path: str,
    input_path_format: str = "mp4",
    H: int = None,
    W: int = None,
    normalize: bool = True,
    max_frames: int = -1,
    also_return_fps: bool = False,
) -> torch.Tensor:
    """Read video or image file and convert to tensor format.

    Args:
        input_path (str): Path to input video/image file
        input_path_format (str): Format of input file (default: "mp4")
        H (int, optional): Height to resize frames to
        W (int, optional): Width to resize frames to
        normalize (bool): Whether to normalize pixel values to [-1,1] (default: True)
        max_frames (int): Maximum number of frames to read (-1 for all frames)
        also_return_fps (bool): Whether to return fps along with frames

    Returns:
        torch.Tensor | tuple: Video tensor in shape [B,C,T,H,W], optionally with fps if requested
    """
    log.debug(f"Reading video from {input_path}")

    loaded_data = load_from_fileobj(input_path, format=input_path_format)
    frames, meta_data = loaded_data
    if input_path.endswith(".png") or input_path.endswith(".jpg") or input_path.endswith(".jpeg"):
        frames = np.array(frames[0])  # HWC, [0,255]
        if frames.shape[-1] > 3:  # RGBA, set the transparent to white
            # Separate the RGB and Alpha channels
            rgb_channels = frames[..., :3]
            alpha_channel = frames[..., 3] / 255.0  # Normalize alpha channel to [0, 1]

            # Create a white background
            white_bg = np.ones_like(rgb_channels) * 255  # White background in RGB

            # Blend the RGB channels with the white background based on the alpha channel
            frames = (rgb_channels * alpha_channel[..., None] + white_bg * (1 - alpha_channel[..., None])).astype(
                np.uint8
            )
        frames = [frames]
        fps = 0
    else:
        fps = int(meta_data.get("fps"))
    if max_frames != -1:
        frames = frames[:max_frames]
    input_tensor = np.stack(frames, axis=0)
    input_tensor = einops.rearrange(input_tensor, "t h w c -> t c h w")
    if normalize:
        input_tensor = input_tensor / 128.0 - 1.0
        input_tensor = torch.from_numpy(input_tensor).bfloat16()  # TCHW
        log.debug(f"Raw data shape: {input_tensor.shape}")
        if H is not None and W is not None:
            input_tensor = transforms_F.resize(
                input_tensor,
                size=(H, W),  # type: ignore
                interpolation=transforms_F.InterpolationMode.BICUBIC,
                antialias=True,
            )
    input_tensor = einops.rearrange(input_tensor, "(b t) c h w -> b c t h w", b=1)
    if normalize:
        input_tensor = input_tensor.to("cuda")
    log.debug(f"Load shape {input_tensor.shape} value {input_tensor.min()}, {input_tensor.max()}")
    if also_return_fps:
        return input_tensor, fps
    return input_tensor


def compute_num_latent_frames(model: DiffusionV2WModel, num_input_frames: int, downsample_factor=8) -> int:
    """This function computes the number of latent frames given the number of input frames.
    Args:
        model (DiffusionV2WModel): video generation model
        num_input_frames (int): number of input frames
        downsample_factor (int): downsample factor for temporal reduce
    Returns:
        int: number of latent frames
    """
    num_latent_frames = (
        num_input_frames
        // model.tokenizer.video_vae.pixel_chunk_duration
        * model.tokenizer.video_vae.latent_chunk_duration
    )
    if num_input_frames % model.tokenizer.video_vae.latent_chunk_duration == 1:
        num_latent_frames += 1
    elif num_input_frames % model.tokenizer.video_vae.latent_chunk_duration > 1:
        assert (
            num_input_frames % model.tokenizer.video_vae.pixel_chunk_duration - 1
        ) % downsample_factor == 0, f"num_input_frames % model.tokenizer.video_vae.pixel_chunk_duration - 1 must be divisible by {downsample_factor}"
        num_latent_frames += (
            1 + (num_input_frames % model.tokenizer.video_vae.pixel_chunk_duration - 1) // downsample_factor
        )

    return num_latent_frames


def create_condition_latent_from_input_frames(
    model: DiffusionV2WModel,
    input_frames: torch.Tensor,
    num_frames_condition: int = 25,
):
    """Create condition latent for video generation from input frames.

    Takes the last num_frames_condition frames from input as conditioning.

    Args:
        model (DiffusionV2WModel): Video generation model
        input_frames (torch.Tensor): Input video tensor [B,C,T,H,W], range [-1,1]
        num_frames_condition (int): Number of frames to use for conditioning

    Returns:
        tuple: (condition_latent, encode_input_frames) where:
            - condition_latent (torch.Tensor): Encoded latent condition [B,C,T,H,W]
            - encode_input_frames (torch.Tensor): Padded input frames used for encoding
    """
    B, C, T, H, W = input_frames.shape
    num_frames_encode = (
        model.tokenizer.pixel_chunk_duration
    )  # (model.state_shape[1] - 1) / model.vae.pixel_chunk_duration + 1
    log.debug(
        f"num_frames_encode not set, set it based on pixel chunk duration and model state shape: {num_frames_encode}"
    )

    log.debug(
        f"Create condition latent from input frames {input_frames.shape}, value {input_frames.min()}, {input_frames.max()}, dtype {input_frames.dtype}"
    )

    assert (
        input_frames.shape[2] >= num_frames_condition
    ), f"input_frames not enough for condition, require at least {num_frames_condition}, get {input_frames.shape[2]}, {input_frames.shape}"
    assert (
        num_frames_encode >= num_frames_condition
    ), f"num_frames_encode should be larger than num_frames_condition, get {num_frames_encode}, {num_frames_condition}"

    # Put the conditioal frames to the begining of the video, and pad the end with zero
    condition_frames = input_frames[:, :, -num_frames_condition:]
    padding_frames = condition_frames.new_zeros(B, C, num_frames_encode - num_frames_condition, H, W)
    encode_input_frames = torch.cat([condition_frames, padding_frames], dim=2)

    log.debug(
        f"create latent with input shape {encode_input_frames.shape} including padding {num_frames_encode - num_frames_condition} at the end"
    )
    if hasattr(model, "n_cameras"):
        encode_input_frames = einops.rearrange(
            encode_input_frames, "(B V) C T H W -> B C (V T) H W", V=model.n_cameras
        )
    latent = model.encode(encode_input_frames)
    return latent, encode_input_frames


def get_condition_latent(
    model: DiffusionV2WModel,
    input_image_or_video_path: str,
    num_input_frames: int = 1,
    state_shape: list[int] = None,
):
    """Get condition latent from input image/video file.

    Args:
        model (DiffusionV2WModel): Video generation model
        input_image_or_video_path (str): Path to conditioning image/video
        num_input_frames (int): Number of input frames for video2world prediction

    Returns:
        tuple: (condition_latent, input_frames) where:
            - condition_latent (torch.Tensor): Encoded latent condition [B,C,T,H,W]
            - input_frames (torch.Tensor): Input frames tensor [B,C,T,H,W]
    """
    if state_shape is None:
        state_shape = model.state_shape
    assert num_input_frames > 0, "num_input_frames must be greater than 0"

    H, W = (
        state_shape[-2] * model.tokenizer.spatial_compression_factor,
        state_shape[-1] * model.tokenizer.spatial_compression_factor,
    )

    input_path_format = input_image_or_video_path.split(".")[-1]
    input_frames = read_video_or_image_into_frames_BCTHW(
        input_image_or_video_path,
        input_path_format=input_path_format,
        H=H,
        W=W,
    )

    condition_latent, _ = create_condition_latent_from_input_frames(model, input_frames, num_input_frames)
    condition_latent = condition_latent.to(torch.bfloat16)

    return condition_latent


def get_condition_latent_multi_camera(
    model: DiffusionMultiCameraV2WModel,
    input_image_or_video_path: str,
    num_input_frames: int = 1,
    state_shape: list[int] = None,
):
    """Get condition latent from input image/video file. This is the function for the multi-camera model where each camera has one latent condition frame.

    Args:
        model (DiffusionMultiCameraV2WModel): Video generation model
        input_image_or_video_path (str): Path to conditioning image/video
        num_input_frames (int): Number of input frames for video2world prediction

    Returns:
        tuple: (condition_latent, input_frames) where:
            - condition_latent (torch.Tensor): Encoded latent condition [B,C,T,H,W]
            - input_frames (torch.Tensor): Input frames tensor [B,C,T,H,W]
    """
    if state_shape is None:
        state_shape = model.state_shape
    assert num_input_frames > 0, "num_input_frames must be greater than 0"

    H, W = (
        state_shape[-2] * model.tokenizer.spatial_compression_factor,
        state_shape[-1] * model.tokenizer.spatial_compression_factor,
    )
    input_path_format = input_image_or_video_path.split(".")[-1]
    input_frames = read_video_or_image_into_frames_BCTHW(
        input_image_or_video_path,
        input_path_format=input_path_format,
        H=H,
        W=W,
    )
    input_frames = einops.rearrange(input_frames, "B C (V T) H W -> (B V) C T H W", V=model.n_cameras)
    condition_latent, _ = create_condition_latent_from_input_frames(model, input_frames, num_input_frames)
    condition_latent = condition_latent.to(torch.bfloat16)

    return condition_latent, einops.rearrange(input_frames, "(B V) C T H W -> B C (V T) H W", V=model.n_cameras)[0]


def check_input_frames(input_path: str, required_frames: int) -> bool:
    """Check if input video/image has sufficient frames.

    Args:
        input_path: Path to input video or image
        required_frames: Number of required frames

    Returns:
        np.ndarray of frames if valid, None if invalid
    """
    if input_path.endswith((".jpg", ".jpeg", ".png")):
        if required_frames > 1:
            log.error(f"Input ({input_path}) is an image but {required_frames} frames are required")
            return False
        return True  # Let the pipeline handle image loading
    # For video input
    try:
        vid = imageio.get_reader(input_path, "ffmpeg")
        frame_count = vid.count_frames()

        if frame_count < required_frames:
            log.error(f"Input video has {frame_count} frames but {required_frames} frames are required")
            return False
        else:
            return True
    except Exception as e:
        log.error(f"Error reading video file {input_path}: {e}")
        return False
