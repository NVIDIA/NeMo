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

from cosmos1.models.diffusion.networks.general_dit_video_conditioned import VideoExtendGeneralDIT
from cosmos1.utils.lazy_config import LazyCall as L
from cosmos1.utils.lazy_config import LazyDict

Cosmos_1_0_Diffusion_Video2World_7B: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /tokenizer": "cosmos_diffusion_tokenizer_res720_comp8x8x8_t121_ver092624"},
            "_self_",
        ],
        model=dict(
            latent_shape=[
                16,
                16,
                88,
                160,
            ],
            conditioner=dict(video_cond_bool=dict()),
            net=L(VideoExtendGeneralDIT)(
                extra_per_block_abs_pos_emb=True,
                rope_h_extrapolation_ratio=1.0,
                rope_w_extrapolation_ratio=1.0,
                rope_t_extrapolation_ratio=2.0,
                extra_per_block_abs_pos_emb_type="learnable",
            ),
        ),
        job=dict(group="Video2World", name="Cosmos_1_0_Diffusion_Video2World_7B"),
    )
)


Cosmos_1_0_Diffusion_Video2World_14B: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_14b"},
            {"override /conditioner": "video_cond"},
            {"override /tokenizer": "cosmos_diffusion_tokenizer_res720_comp8x8x8_t121_ver092624"},
            "_self_",
        ],
        model=dict(
            latent_shape=[
                16,
                16,
                88,
                160,
            ],
            conditioner=dict(video_cond_bool=dict()),
            net=L(VideoExtendGeneralDIT)(
                extra_per_block_abs_pos_emb=True,
                rope_h_extrapolation_ratio=2.0,
                rope_t_extrapolation_ratio=2.0,
                rope_w_extrapolation_ratio=2.0,
                extra_h_extrapolation_ratio=2.0,
                extra_t_extrapolation_ratio=2.0,
                extra_w_extrapolation_ratio=2.0,
                extra_per_block_abs_pos_emb_type="learnable",
            ),
        ),
        job=dict(group="Video2World", name="Cosmos_1_0_Diffusion_Video2World_14B"),
    )
)

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=Cosmos_1_0_Diffusion_Video2World_7B["job"]["name"],
    node=Cosmos_1_0_Diffusion_Video2World_7B,
)

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=Cosmos_1_0_Diffusion_Video2World_14B["job"]["name"],
    node=Cosmos_1_0_Diffusion_Video2World_14B,
)
