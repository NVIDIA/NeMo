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

from typing import Any, List, Union

import attrs

from cosmos1.models.autoregressive.configs.base.model import ModelConfig, TokenizerConfig


@attrs.define(slots=False)
class DataShapeConfig:
    latent_shape: list = []
    num_video_frames: Union[None, int] = None
    height: Union[None, int] = None
    width: Union[None, int] = None


@attrs.define(slots=False)
class SamplingConfig:
    """
    Sampling config
    Args:
        temperature (float): Temperature value for controlling randomness in sampling. Defaults to 0.6.
        top_p (float): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
        logprobs (bool): Flag indicating whether to compute token log probabilities. Defaults to False.
        echo (bool): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

    """

    temperature: float = 0.6
    top_k: int = None
    top_p: float = 0.9
    compile_prefill: bool = False
    compile_sampling: bool = True
    logprobs: bool = False
    echo: bool = False


@attrs.define(slots=False)
class DiffusionDecoderSamplingConfig:
    """
    Diffusion decoder sampling config
    Args:
        guidance (float): Guidance scale for the diffusion process. Controls how much the model follows the conditioning. Defaults to 0.8.
        sigma_min (float): Minimum noise level for the diffusion process. Defaults to 0.02.
        sigma (float): Initial noise level for the diffusion process. Defaults to 8.
        num_steps (int): Number of denoising steps to perform. Defaults to 35.
        overlap (int): Number of overlapping frames between video chunks during processing. Defaults to 2.
        continuous_tokenizer_channel (int): Number of channels in the continuous tokenizer of diffusion decoder. Defaults to 16.
        continuous_tokenizer_spatial_compression_ratio (int): Spatial compression ratio for the continuous tokenizer of diffusion decoder. Defaults to 8.
        dd_train_num_video_frames (int): Number of video frames used during training for diffusion decoder. Defaults to 57.
    """

    guidance: float = 1.8
    sigma_min: float = 0.02
    sigma: float = 8
    num_steps: int = 15
    overlap: int = 2
    continuous_tokenizer_channel = 16
    continuous_tokenizer_spatial_compression_ratio = 8
    dd_train_num_video_frames: int = 57
    max_iter: int = 99
    fps: int = 24


@attrs.define(slots=False)
class InferenceConfig:
    """
    Inference config
    Args:
        model_config (ModelConfig): Model config
        tokenizer_config (TokenizerConfig): Tokenizer config
        ckpt_path (str): Path to the checkpoint
        latent_shape (list): Shape of the latent
    """

    model_config: ModelConfig = None
    tokenizer_config: TokenizerConfig = None
    ckpt_path: str = ""
    data_shape_config: DataShapeConfig = None

    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_val": None},
            {"data_shape_config": "video_shape_as_model_config"},
            {"eval_job": None},
        ]
    )
