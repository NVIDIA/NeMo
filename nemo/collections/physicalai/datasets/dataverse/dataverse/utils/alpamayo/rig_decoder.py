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

import dataverse.utils.alpamayo.transformation as transformation


def decode_rig_info(rig) -> dict:
    """Decode rig information from json bytes."""
    # rig = json.loads(rig_json)
    sensor_info = transformation.parse_rig_sensors_from_dict(rig)
    vehicle_info = rig["rig"]["vehicle"]
    result = {"_".join(name.split(":")): _ for name, _ in sensor_info.items()}
    result["vehicle"] = vehicle_info
    return result
