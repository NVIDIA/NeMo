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

# Follwoing are temporarily adapted and modified from:
# https://github.com/meta-llama/llama-stack/tree/main/llama_stack/models/llama

from pydantic import BaseModel


class Size(BaseModel):
    height: int
    width: int


class VisionArgs(BaseModel):
    image_size: Size
    patch_size: Size

    # parameters for the encoder transformer
    dim: int
    n_layers: int
    n_heads: int
    mlp_ratio: float
    output_dim: int

    pixel_shuffle_ratio: float

    @classmethod
    def preprocess_values(cls, data: dict) -> dict:
        data = data.copy()
        if "image_size" not in data:
            data["image_size"] = Size(height=data["image_height"], width=data["image_width"])
        if "patch_size" not in data:
            data["patch_size"] = Size(height=data["patch_height"], width=data["patch_width"])
        if "pixel_shuffle_ratio" not in data:
            data["pixel_shuffle_ratio"] = data["ps_ratio"]
        if "dim" not in data:
            data |= {
                "dim": 1408,
                "n_layers": 34,
                "n_heads": 16,
                "mlp_ratio": 4.0,
                "output_dim": 4096,
            }
        return data
