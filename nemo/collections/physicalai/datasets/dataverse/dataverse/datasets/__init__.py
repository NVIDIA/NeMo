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

# pylint: disable=C0115,C0116,C0301

from enum import Enum


class DataField(Enum):
    # [B, C, H, W], float32, RGB image ranges from 0 to 1.
    IMAGE_RGB = "image_rgb"
    # [B, 4, 4], float32, camera-to-world transformation matrix.
    CAMERA_C2W_TRANSFORM = "camera_c2w_transform"
    # [B, 4], float32, OpenCV pinhole intrinsics represented as [fx, fy, cx, cy].
    CAMERA_INTRINSICS = "camera_intrinsics"
    # list of captions of size B.
    CAPTION = "caption"
    # [B, H, W], float32, depth map in metric scale.
    METRIC_DEPTH = "metric_depth"
    # [B, H, W], uint8, instance mask (0 is background).
    DYNAMIC_INSTANCE_MASK = "dynamic_instance_mask"
    # [B, H, W], float32, backward flow from this frame to previous frame.
    BACKWARD_FLOW = "backward_flow"
    # [B, H, W, 3], float32, ray direction (assume no motion/RS).
    RAY_DIRECTION = "ray_direction"
    # TODO [Add description]
    OBJECT_BBOX = "object_bbox"
    # TODO [Add description] a list of float32 point cloud.
    POINT_CLOUD = "point_cloud"
    # [B, N, (3 + 3x3)], N future positions. For the last dim,
    # the first 3 are xyz locations, and tha last 9 are rots
    # B corresponds to the number of timestamps for the base camera type
    TRAJECTORY = 'trajectory'
