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

from typing import Final

CROSS_LEFT_CAMERA_NAME: Final = "camera_cross_left_120fov"
CROSS_RIGHT_CAMERA_NAME: Final = "camera_cross_right_120fov"
FRONT_TELE_CAMERA_NAME: Final = "camera_front_tele_30fov"
FRONT_WIDE_CAMERA_NAME: Final = "camera_front_wide_120fov"
REAR_LEFT_CAMERA_NAME: Final = "camera_rear_left_70fov"
REAR_RIGHT_CAMERA_NAME: Final = "camera_rear_right_70fov"
REAR_TELE_CAMERA_NAME: Final = "camera_rear_tele_30fov"

# Camera indices will be used for computing camera embeddings.
CAMERA_NAMES_TO_INDICES = {
    CROSS_LEFT_CAMERA_NAME: 0,
    FRONT_WIDE_CAMERA_NAME: 1,
    CROSS_RIGHT_CAMERA_NAME: 2,
    REAR_LEFT_CAMERA_NAME: 3,
    REAR_TELE_CAMERA_NAME: 4,
    REAR_RIGHT_CAMERA_NAME: 5,
    FRONT_TELE_CAMERA_NAME: 6,
}
CAMERA_NAMES = tuple(name for name in CAMERA_NAMES_TO_INDICES)
CAMERA_INDICES_TO_NAMES = {idx: name for name, idx in CAMERA_NAMES_TO_INDICES.items()}

# fps for v2 data.
VIDEO_FPS = 30

# minimum start index.
# NOTE V2 data might have issue in the first few frames, set a minimum to exclude them.
MIN_START_INDEX = 3

# Roughly-estimated car dimensions (length, width, height) in meters.
# For full accuracy, the actual ego-vehicle dimensions for each clip can be found
# in the rig.json file, as rig["rig"]["vehicle"]["value"]["body"].
EGO_VEHICLE_LWH = (4.0, 3.0, 2.0)
# position of ego coordinate frame along the length of the car
# (distance of rear axle from back bumper / the length of the car)
# true value can be computed from rig.json file using
# -rig["rig"]["vehicle"]["value"]["body"]["boundingBoxPosition"][0] / ego_length
# we assume it is centered along the width, and so far are not using the z offset
EGO_VEHICLE_LENGTH_OFFSET = 0.21

# Mapping between raw object sub-categories loaded from the data and grouped object
# class categories used for training the detector
CLASS_TO_SUBCATEGORY_INV = {
    "automobile": "Car",
    "trailer": "Car",
    "truck": "Truck",
    "heavy_truck": "Truck",
    "bus": "Truck",
    "train_or_tram_car": "Truck",
    "other_vehicle": "Truck",
    "pedestrian": "Pedestrian",
    "person": "Pedestrian",
    "person_group": "Pedestrian",
    "bicycle": "Cyclist",
    "stroller": "Cyclist",
    "bicycle_with_rider": "Cyclist",
    "motorcycle_with_rider": "Cyclist",
    "motorcycle": "Cyclist",
    "rider": "Cyclist",
    "cycle": "Others",
    "protruding_object": "Others",
    "animal": "Others",
    "unknown": "Others",
}
CLASS_TO_SUBCATEGORY = {
    "Car": "automobile",
    "Truck": "truck",
    "Pedestrian": "pedestrian",
    "Cyclist": "bicycle",
    "Others": "unknown",
}
CLASSES = list(CLASS_TO_SUBCATEGORY.keys())
CLASSES_SUB = list(CLASS_TO_SUBCATEGORY_INV.keys())

LIDAR_NAME: Final = "lidar_gt_top_p128"
"""Identifier for the LiDAR sensor in the dataset."""

ALPACKAGES_BUCKET: Final = "alpackages"
"""The name of the S3/SwiftStack bucket/container where alpackages are stored."""

S3_REGION: Final = "us-east-1"
"""The region of the S3/SwiftStack bucket/container."""

S3_ENDPOINT_URL: Final = "https://pdx.s8k.io"
"""The endpoint URL for the S3/SwiftStack bucket/container."""
