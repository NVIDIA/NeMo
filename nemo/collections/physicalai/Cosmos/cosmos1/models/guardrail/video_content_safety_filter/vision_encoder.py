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

import torch
from PIL import Image
from transformers import SiglipModel, SiglipProcessor

DEFAULT_CHECKPOINT_DIR = "checkpoints/Cosmos-1.0-Guardrail/video_content_safety_filter"


class SigLIPEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
    ) -> None:
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.dtype = dtype
        self.model = SiglipModel.from_pretrained(model_name, cache_dir=self.checkpoint_dir)
        self.processor = SiglipProcessor.from_pretrained(model_name, cache_dir=self.checkpoint_dir)
        self.model.to(self.device, dtype=self.dtype).eval()

    @torch.inference_mode()
    def encode_image(self, input_img: Image.Image) -> torch.Tensor:
        """Encode an image into a feature vector."""
        with torch.no_grad():
            inputs = self.processor(images=input_img, return_tensors="pt").to(self.device, dtype=self.dtype)
            image_features = self.model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
