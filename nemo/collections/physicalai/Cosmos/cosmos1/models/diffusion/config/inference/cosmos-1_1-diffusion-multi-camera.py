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

from cosmos1.models.diffusion.networks.general_dit_multi_camera import MultiCameraVideoExtendGeneralDIT
from cosmos1.utils.lazy_config import LazyCall as L
from cosmos1.utils.lazy_config import LazyDict
from hydra.core.config_store import ConfigStore

Cosmos_1_1_Diffusion_Multi_Camera_Text2World_7B: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/Cosmos_1_0_Diffusion_Text2World_7B",
            {"override /net": "faditv2_multicam_7b"},
            {"override /conditioner": "add_fps_image_size_padding_mask_frame_repeat"},
            "_self_",
        ],
        job=dict(
            group="Text2World",
            name="Cosmos_1_1_Diffusion_Multi_Camera_Text2World_7B",
        ),
        model=dict(
            latent_shape=[
                16,
                16,
                88,
                160,
            ],
            tokenizer=dict(
                video_vae=dict(
                    pixel_chunk_duration=57,
                )
            ),
        ),
    )
)

Cosmos_1_1_Diffusion_Multi_Camera_Video2World_7B: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/Cosmos_1_1_Diffusion_Multi_Camera_Text2World_7B",
            {"override /conditioner": "video_cond_frame_repeat"},
            "_self_",
        ],
        job=dict(
            group="Text2World",
            name="Cosmos_1_1_Diffusion_Multi_Camera_Video2World_7B",
        ),
        model=dict(
            latent_shape=[
                16,
                16,
                88,
                160,
            ],
            net=L(MultiCameraVideoExtendGeneralDIT)(
                n_cameras=6,
                camera_condition_dim=6,
                add_repeat_frame_embedding=True,
            ),
            conditioner=dict(video_cond_bool=dict()),
        ),
    )
)


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=Cosmos_1_1_Diffusion_Multi_Camera_Text2World_7B["job"]["name"],
    node=Cosmos_1_1_Diffusion_Multi_Camera_Text2World_7B,
)


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=Cosmos_1_1_Diffusion_Multi_Camera_Video2World_7B["job"]["name"],
    node=Cosmos_1_1_Diffusion_Multi_Camera_Video2World_7B,
)
