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

import dataverse.utils.alpamayo.constants as constants
import dataverse.utils.alpamayo.ndas_camera_model as ndas_camera_model


def decode_intrinsic(rig_info: dict, camera_indices: list[int], raise_on_unsupported: bool = False):
    """Decode intrinsics for those cameras in camera_indices."""
    outputs = {}
    for name, sensor in rig_info.items():
        # if load with RANDOM, we will load all camera intrins to make the batching happy
        name = name.replace('.mp4', '')
        if name in constants.CAMERA_NAMES:
            cam_id = constants.CAMERA_NAMES_TO_INDICES[name]
            if cam_id in camera_indices:
                if sensor["properties"]["polynomial-type"] != "pixeldistance-to-angle":
                    if raise_on_unsupported:
                        raise ValueError(f"{name} has an unsupported polynomial-type!")
                    else:
                        outputs[name] = None
                else:
                    outputs[name] = ndas_camera_model.FThetaCamera.from_dict(sensor)
    return outputs
