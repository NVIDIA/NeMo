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

import gc
import importlib
import math
import os
from typing import List

import torch
import torchvision
from cosmos1.models.autoregressive.configs.inference.inference_config import DiffusionDecoderSamplingConfig
from cosmos1.models.autoregressive.diffusion_decoder.inference import diffusion_decoder_process_tokens
from cosmos1.models.autoregressive.diffusion_decoder.model import LatentDiffusionDecoderModel
from cosmos1.models.diffusion.inference.inference_utils import (
    load_network_model,
    load_tokenizer_model,
    skip_init_linear,
)
from cosmos1.utils import log
from cosmos1.utils.config_helper import get_config_module, override
from huggingface_hub import snapshot_download

DATA_RESOLUTION_DEFAULT = [640, 1024]
NUM_CONTEXT_FRAMES_DEFAULT = 33


def resize_input(video: torch.Tensor, resolution: list[int]):
    r"""
    Function to perform aspect ratio preserving resizing and center cropping.
    This is needed to make the video into target resolution.
    Args:
        video (torch.Tensor): Input video tensor
        resolution (list[int]): Data resolution
    Returns:
        Cropped video
    """

    orig_h, orig_w = video.shape[2], video.shape[3]
    target_h, target_w = resolution

    scaling_ratio = max((target_w / orig_w), (target_h / orig_h))
    resizing_shape = (int(math.ceil(scaling_ratio * orig_h)), int(math.ceil(scaling_ratio * orig_w)))
    video_resized = torchvision.transforms.functional.resize(video, resizing_shape)
    video_cropped = torchvision.transforms.functional.center_crop(video_resized, resolution)
    return video_cropped


def read_input_videos(
    input_video: str,
    data_resolution: list[int] = DATA_RESOLUTION_DEFAULT,
    num_frames: int = NUM_CONTEXT_FRAMES_DEFAULT,
) -> torch.tensor:
    """Utility to read the input video and return a torch tensor

    Args:
        input_video (str): A path to .mp4 file
        data_resolution (list, optional): The . Defaults to [640, 1024].
        num_frames (int, optional): The number of frames to read.

    Returns:
        A torch tensor of the video
    """
    video, _, _ = torchvision.io.read_video(input_video)
    video = video.float() / 255.0
    video = video * 2 - 1
    if video.shape[0] > num_frames:
        video = video[0:num_frames, :, :, :]
    else:
        log.info(f"Video doesn't have {num_frames} frames. Padding the video with the last frame.")
        # Pad the video
        nframes_in_video = video.shape[0]
        video = torch.cat(
            (video, video[-1, :, :, :].unsqueeze(0).repeat(num_frames - nframes_in_video, 1, 1, 1)),
            dim=0,
        )

    video = video[0:num_frames, :, :, :]
    video = video.permute(0, 3, 1, 2)
    video = resize_input(video, data_resolution)
    return video.transpose(0, 1).unsqueeze(0)


def run_diffusion_decoder_model(indices_tensor_cur_batch: List[torch.Tensor], out_videos_cur_batch):
    """Run a 7b diffusion model to enhance generation output

    Args:
        indices_tensor_cur_batch (List[torch.Tensor]): The index tensor(i.e) prompt + generation tokens
        out_videos_cur_batch (torch.Tensor): The output decoded video of shape [bs, 3, 33, 640, 1024]
    """
    diffusion_decoder_ckpt_path = snapshot_download("nvidia/Cosmos-1.0-Diffusion-7B-Decoder-DV8x16x16ToCV8x8x8")
    dd_tokenizer_dir = snapshot_download("nvidia/Cosmos-1.0-Tokenizer-CV8x8x8")
    tokenizer_corruptor_dir = snapshot_download("nvidia/Cosmos-1.0-Tokenizer-DV8x16x16")

    diffusion_decoder_model = load_model_by_config(
        config_job_name="DD_FT_7Bv1_003_002_tokenizer888_spatch2_discrete_cond_on_token",
        config_file="cosmos1/models/autoregressive/diffusion_decoder/config/config_latent_diffusion_decoder.py",
        model_class=LatentDiffusionDecoderModel,
        encoder_path=os.path.join(tokenizer_corruptor_dir, "encoder.jit"),
        decoder_path=os.path.join(tokenizer_corruptor_dir, "decoder.jit"),
    )
    load_network_model(diffusion_decoder_model, os.path.join(diffusion_decoder_ckpt_path, "model.pt"))
    load_tokenizer_model(diffusion_decoder_model, dd_tokenizer_dir)

    generic_prompt = dict()
    aux_vars = torch.load(os.path.join(diffusion_decoder_ckpt_path, "aux_vars.pt"), weights_only=True)
    generic_prompt["context"] = aux_vars["context"].cuda()
    generic_prompt["context_mask"] = aux_vars["context_mask"].cuda()

    output_video = diffusion_decoder_process_tokens(
        model=diffusion_decoder_model,
        indices_tensor=indices_tensor_cur_batch,
        dd_sampling_config=DiffusionDecoderSamplingConfig(),
        original_video_example=out_videos_cur_batch[0],
        t5_emb_batch=[generic_prompt["context"]],
    )

    del diffusion_decoder_model
    gc.collect()
    torch.cuda.empty_cache()

    return output_video


def load_model_by_config(
    config_job_name,
    config_file="projects/cosmos_video/config/config.py",
    model_class=LatentDiffusionDecoderModel,
    encoder_path=None,
    decoder_path=None,
):
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()

    config = override(config, ["--", f"experiment={config_job_name}"])

    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    if encoder_path:
        config.model.tokenizer_corruptor["enc_fp"] = encoder_path
    if decoder_path:
        config.model.tokenizer_corruptor["dec_fp"] = decoder_path
    # Initialize model
    with skip_init_linear():
        model = model_class(config.model)
    return model
