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

import attrs
import torch
import torch.nn as nn

from cosmos1.utils.config import make_freezable


@make_freezable
@attrs.define(slots=False)
class ModelConfig:
    input_size: int = 1152
    num_classes: int = 7


class SafetyClassifier(nn.Module):
    def __init__(self, input_size: int = 1024, num_classes: int = 2):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes),
            # Note: No activation function here; CrossEntropyLoss expects raw logits
        )

    def forward(self, x):
        return self.layers(x)


class VideoSafetyModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.network = SafetyClassifier(input_size=config.input_size, num_classes=self.num_classes)

    @torch.inference_mode()
    def forward(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logits = self.network(data_batch["data"].cuda())
        return {"logits": logits}
