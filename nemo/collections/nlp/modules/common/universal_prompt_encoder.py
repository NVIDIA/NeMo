# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from torch import nn

from nemo.collections.nlp.modules.common.megatron.megatron_perceiver_encoders import MegatronPerceiverEncoderModule
from nemo.core.classes import Exportable, NeuralModule


class UniversalPromptEncoder(NeuralModule, Exportable):
    def __init__(
        self, cfg, output_dim,
    ):
        """
        """
        super().__init__()
        self.encoder = MegatronPerceiverEncoderModule(**cfg, parent_model_type=None)
        self.hidden = self.encoder.hidden_size
        self.input_linear = nn.Linear(output_dim, self.hidden)
        self.output_linear = nn.Linear(self.hidden, output_dim)
        # input_adaptor = nn.Linear(self.hidden_size, output_size)

    def forward(self, input_prompt, mask) -> torch.Tensor:
        input_prompt = self.input_linear(input_prompt)
        hidden = self.encoder.forward(input_prompt, mask)
        hidden = self.output_linear(hidden)
        return hidden
