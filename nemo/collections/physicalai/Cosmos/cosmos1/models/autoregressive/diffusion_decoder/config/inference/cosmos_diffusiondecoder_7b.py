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

from cosmos1.models.autoregressive.diffusion_decoder.network import DiffusionDecoderGeneralDIT
from cosmos1.utils.lazy_config import LazyCall as L
from cosmos1.utils.lazy_config import LazyDict

num_frames = 57
Cosmos_DiffusionDecoder_7B_INFERENCE_ONLY: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /tokenizer": "cosmos_video_tokenizer_res720_comp8x8x8_t121_ver092624"},
            {"override /conditioner": "video_latent_diffusion_decoder_cond"},
            {"override /tokenizer_corruptor": "cosmos_video_discrete_tokenizer_res720_comp8x16x16_t49_ver110224"},
            "_self_",
        ],
        job=dict(
            group="diffusion_deocder_FT_7Bv1_001",
            name="DD_FT_7Bv1_003_002_tokenizer888_spatch2_discrete_cond_on_token",
        ),
        model=dict(
            diffusion_decoder_cond_sigma_low=0.0,
            diffusion_decoder_cond_sigma_high=0.0,
            diffusion_decoder_corrupt_prob=0.0,
            condition_on_tokenizer_corruptor_token=True,
            latent_shape=[
                16,
                num_frames,
                88,
                160,
            ],
            tokenizer_corruptor=dict(
                pixel_chunk_duration=num_frames,
                latent_chunk_duration=1 + (num_frames - 1) // 8,
            ),
            net=L(DiffusionDecoderGeneralDIT)(
                diffusion_decoder_condition_on_sigma=False,
                max_img_h=240,
                max_img_w=240,
                rope_h_extrapolation_ratio=1.5,
                rope_w_extrapolation_ratio=1.5,
                rope_t_extrapolation_ratio=1,
                block_x_format="THWBD",
                is_diffusion_decoder=True,
                patch_spatial=2,
                diffusion_decoder_condition_on_token=True,
                diffusion_decoder_token_condition_voc_size=64000,
                diffusion_decoder_token_condition_dim=32,
            ),
            tokenizer=dict(
                video_vae=dict(
                    pixel_chunk_duration=num_frames,
                )
            ),
            conditioner=dict(
                latent_condition=dict(
                    dropout_rate=0.2,
                )
            ),
        ),
    )
)

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=Cosmos_DiffusionDecoder_7B_INFERENCE_ONLY["job"]["name"],
    node=Cosmos_DiffusionDecoder_7B_INFERENCE_ONLY,
)
