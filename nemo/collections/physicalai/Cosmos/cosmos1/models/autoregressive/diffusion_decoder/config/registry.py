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

from hydra.core.config_store import ConfigStore

from cosmos1.models.autoregressive.diffusion_decoder.config.base.conditioner import (
    VideoLatentDiffusionDecoderConditionerConfig,
)
from cosmos1.models.autoregressive.tokenizer.discrete_video import DiscreteVideoFSQJITTokenizer
from cosmos1.models.diffusion.module.pretrained_vae import JITVAE, JointImageVideoSharedJITTokenizer, VideoJITTokenizer
from cosmos1.utils.lazy_config import LazyCall as L


def get_cosmos_video_discrete_tokenizer_comp8x16x16(
    resolution: str,
    chunk_duration: int,
    checkpoint_path: str,
):
    assert resolution in ["720"]

    pixel_chunk_duration = chunk_duration
    temporal_compression_factor = 8
    spatial_compression_factor = 16

    return L(DiscreteVideoFSQJITTokenizer)(
        enc_fp=checkpoint_path.replace(".jit", "encoder.jit"),
        dec_fp=checkpoint_path.replace(".jit", "decoder.jit"),
        name="discrete_video_fsq",
        latent_ch=6,
        is_bf16=True,
        pixel_chunk_duration=pixel_chunk_duration,
        latent_chunk_duration=1 + (pixel_chunk_duration - 1) // temporal_compression_factor,
        max_enc_batch_size=8,
        max_dec_batch_size=4,
        levels=[8, 8, 8, 5, 5, 5],
        compression_ratio=[temporal_compression_factor, spatial_compression_factor, spatial_compression_factor],
    )


def get_cosmos_video_tokenizer_comp8x8x8(resolution: str, chunk_duration: int, checkpoint_path=None):
    pixel_chunk_duration = chunk_duration
    temporal_compression_factor = 8
    spatial_compression_factor = 8

    return L(JointImageVideoSharedJITTokenizer)(
        video_vae=L(VideoJITTokenizer)(
            name="cosmos_1_0_diffusion_tokenizer",
            latent_ch=16,
            is_bf16=True,
            pixel_chunk_duration=pixel_chunk_duration,
            temporal_compression_factor=temporal_compression_factor,
            spatial_compression_factor=spatial_compression_factor,
            spatial_resolution=resolution,
        ),
        image_vae=L(JITVAE)(
            name="cosmos_1_0_diffusion_tokenizer",
            latent_ch=16,
            is_image=False,
            is_bf16=True,
        ),
        name="cosmos_diffusion_tokenizer_res720_comp8x8x8_t121_ver092624",
        latent_ch=16,
    )


def register_tokenizer(cs):
    cs.store(
        group="tokenizer",
        package="model.tokenizer",
        name="cosmos_video_tokenizer_res720_comp8x8x8_t121_ver092624",
        node=get_cosmos_video_tokenizer_comp8x8x8(
            resolution="720",
            chunk_duration=121,
            checkpoint_path="checkpoints/Cosmos-1.0-Tokenizer-CV8x8x8/.jit",
        ),
    )


def register_corruptor(cs):
    cs.store(
        group="tokenizer_corruptor",
        package="model.tokenizer_corruptor",
        name="cosmos_video_discrete_tokenizer_res720_comp8x16x16_t49_ver110224",
        node=get_cosmos_video_discrete_tokenizer_comp8x16x16(
            resolution="720",
            chunk_duration=49,
            checkpoint_path="checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16/.jit",
        ),
    )


def register_conditioner(cs):
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="video_latent_diffusion_decoder_cond",
        node=VideoLatentDiffusionDecoderConditionerConfig,
    )


def register_configs():
    cs = ConfigStore.instance()

    register_conditioner(cs)
    register_corruptor(cs)
    register_tokenizer(cs)
