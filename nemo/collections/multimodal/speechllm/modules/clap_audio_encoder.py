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
        window_length_in_secs: float = 0.32,
        window_stride_in_secs: float = 0.2,
        output_dim=None,
        hidden_dim: int = 512,
        num_layers: int = 1,
        activation: str = "relu",
        sample_rate: int = 16000,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.clap_model = ClapAudioModelWithProjection.from_pretrained(pretrained_model)
        self.clap_processor = ClapProcessor.from_pretrained(pretrained_model)
        input_dim = self.clap_model.config.projection_dim
        self.output_dim = output_dim
        self.window_length_in_secs = window_length_in_secs
        self.window_stride_in_secs = window_stride_in_secs
        self.segment_length = int(self.window_length_in_secs * sample_rate)
        self.shift_length = int(self.window_stride_in_secs * sample_rate)
        self.sample_rate = sample_rate

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

    def forward_clap(self, audio_signal):
        audio_signal_48k = audio_signal.cpu().repeat_interleave(3, dim=1)
        import ipdb

        ipdb.set_trace()
        inputs = self.clap_processor(audios=audio_signal_48k, return_tensors="pt", sampling_rate=48000)
        import ipdb

        ipdb.set_trace()
        outputs = self.clap_model(**inputs)
        audio_embeds = outputs.audio_embeds  # [batch_size, clap_dim]
        embeds = self.mlp(audio_embeds)
        return embeds.unsqueeze(-1)  # [batch_size, output_dim, 1]

    def forward(self, audio_signal, length):
        """
        Args:
            audio_signal: [batch_size, num_samples]
            length: [batch_size]
        Returns:
            outputs: [batch_size, output_dim, num_segments]
            outputs_lengths: [batch_size]
        """
        outputs = []
        outputs_lengths = torch.zeros(audio_signal.size(0), dtype=torch.int64, device=audio_signal.device)
        idx = 0
        while idx < audio_signal.size(1):
            audio_segment = audio_signal[:, idx : idx + self.segment_length]
            audio_segment_length = torch.ones_like(outputs_lengths) * (idx < length).long()
            outputs_lengths += audio_segment_length
            if audio_segment.size(1) < self.segment_length:
                audio_segment = torch.cat(
                    [
                        audio_segment,
                        torch.zeros(
                            audio_segment.size(0),
                            self.segment_length - audio_segment.size(1),
                            device=audio_segment.device,
                        ),
                    ],
                    dim=1,
                )
            embeds = self.forward_clap(audio_segment)
            outputs.append(embeds)
            idx += self.shift_length
        outputs = torch.cat(outputs, dim=-1)
        return outputs, outputs_lengths
