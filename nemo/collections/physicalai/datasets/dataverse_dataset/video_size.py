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

VIDEO_RES_SIZE_INFO: dict[str, tuple[int, int]] = {
    "1080": {  # 1080p doesn't have 1:1
        "1,1": (1024, 1024),
        "4,3": (1440, 1056),
        "3,4": (1056, 1440),
        "16,9": (1920, 1056),
        "9,16": (1056, 1920),
    },
    # 1024; the video format does not support it, but here we match it with image resolution
    "1024": {"1,1": (1024, 1024), "4,3": (1280, 1024), "3,4": (1024, 1280), "16,9": (1280, 768), "9,16": (768, 1280)},
    "720": {"1,1": (960, 960), "4,3": (960, 704), "3,4": (704, 960), "16,9": (1280, 704), "9,16": (704, 1280)},
    "512": {"1,1": (512, 512), "4,3": (640, 512), "3,4": (512, 640), "16,9": (640, 384), "9,16": (384, 640)},
    "256": {
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (320, 192),
        "9,16": (192, 320),
    },
}
