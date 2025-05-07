# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from omegaconf import DictConfig
from platformdirs import user_cache_path

dataset_path = str(user_cache_path("AV-V2.2"))

DATAVERSE_CONFIG = dict()
tar_dirs_training = [
    f"{dataset_path}/trainv2-2-chunk-00",
    f"{dataset_path}/trainv2-2-chunk-01",
    f"{dataset_path}/trainv2-2-chunk-02",
    f"{dataset_path}/trainv2-2-chunk-03",
    f"{dataset_path}/trainv2-2-chunk-04",
    f"{dataset_path}/trainv2-2-chunk-05",
    f"{dataset_path}/trainv2-2-chunk-06",
    f"{dataset_path}/trainv2-2-chunk-07",
    f"{dataset_path}/trainv2-2-chunk-08",
    f"{dataset_path}/trainv2-2-chunk-09",
    f"{dataset_path}/trainv2-2-chunk-10",
    f"{dataset_path}/trainv2-2-chunk-11",
    f"{dataset_path}/trainv2-2-chunk-12",
    f"{dataset_path}/trainv2-2-chunk-13",
    f"{dataset_path}/trainv2-2-chunk-14",
    f"{dataset_path}/trainv2-2-chunk-15",
]

tar_dirs_training_full = tar_dirs_training + [
    f"{dataset_path}/trainv2-2-chunk-16",
    f"{dataset_path}/trainv2-2-chunk-17",
    f"{dataset_path}/trainv2-2-chunk-18",
    f"{dataset_path}/trainv2-2-chunk-19",
    f"{dataset_path}/trainv2-2-chunk-20",
    f"{dataset_path}/trainv2-2-chunk-21",
    f"{dataset_path}/trainv2-2-chunk-22",
]
cmeare_groups = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_tele_30fov",
]
cmeare_6_group = cmeare_groups + ["camera_rear_left_70fov", "camera_rear_right_70fov"]


DATAVERSE_CONFIG["alpamayo_v2_traj_qwen_24fps_6_cameras_frame_repeat"] = {
    "dataset_cfg": DictConfig(
        {
            "target": "nemo.collections.physicalai.datasets.dataverse.dataverse.datasets.cosmos_av_w_traj.CosmosAV",
            "params": {
                "tar_dirs": tar_dirs_training,
                "uuid_dirs": f"{dataset_path}/uuid",
                "t5_dirs": f"{dataset_path}/alpamayo_caption_t5/qwen_t5_tars/",
                "probe_tar": True,
                "camkeys": cmeare_6_group,
                "rectify": {"enabled": False},
                "use_hq_data": True,
                "decode_traj": True,
            },
        }
    ),
    "sample_n_frames": 57,
    "sample_size": [480, 848],
    "fps": 24,
    "load_video": True,
    "load_frame_repeat": True,
    "load_trajectory": True,
}
