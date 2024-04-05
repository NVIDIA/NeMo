# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import torch
import torch.nn as nn
from transformers import ClapAudioModelWithProjection, ClapProcessor

from nemo.collections.common.parts.multi_layer_perceptron import MultiLayerPerceptron as MLP
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AccessMixin
from nemo.core.classes.module import NeuralModule


class Flatten(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.flatten(x, self.dim)


class CLAPAudioEncoder(NeuralModule, Exportable, AccessMixin):
    def __init__(
        self,
        pretrained_model: str = "laion/clap-htsat-fused",
        output_dim=None,
        hidden_dim: int = 512,
        num_layers: int = 1,
        activation: str = "relu",
    ):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.clap_model = ClapAudioModelWithProjection.from_pretrained(pretrained_model)
        self.clap_processor = ClapProcessor.from_pretrained(pretrained_model)
        input_dim = self.clap_model.config.projection_dim
        self.output_dim = output_dim
        if output_dim is None:
            self.mlp = nn.Identity()
            self.output_dim = input_dim
        else:
            self.mlp = MLP(
                input_size=input_dim,
                output_size=output_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                activation=activation,
            )

    def forward(self, audio_signal, audio_signal_length):
        inputs = self.clap_processor(audios=audio_signal, return_tensors="pt")
        outputs = self.clap_model(**inputs)
        audio_embeds = outputs.audio_embeds  # [batch_size, clap_dim]
        embeds = self.mlp(audio_embeds)
        return embeds.unsqueeze(-1)  # [batch_size, output_dim, 1]
