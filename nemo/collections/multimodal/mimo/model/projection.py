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
# limitations under the License. Add some stuff

from dataclasses import dataclass, field

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import init_method_normal


class ImageOutputProjectionModule(MegatronModule):
    def __init__(self, encoder_spec, decoder_spec):
        super().__init__()

        # inp linear layer

        # transformer encoder layer with no bidirectional mask

        # transformer decoder layer with self and cross attention

        # learnable query embeddings input to transformer decoder

        # oup linear layer


class ImageOutputProjectionPoolingHead(TransformerLayer):
    """Multihead Attention Pooling."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        num_query_token=77,
    ):
        super().__init__(config, submodules)

        self.probe = torch.nn.Parameter(torch.randn(num_query_token, 1, config.hidden_size))
        self.output_projection = build_module(
            submodules.output_linear_layer,
            input_size=4096,
            output_size=1024,
            config=config,
            bias=True,
            gather_output=True,
            skip_bias_add=False,
            is_expert=False,
            init_method=init_method_normal(0.02),
        )

    def forward(self, hidden_state):

        batch_size = hidden_state.shape[0]

        # [s, b, h]
        probe = self.probe.repeat(1, batch_size, 1)
        hidden_state = hidden_state.transpose(0, 1)
        hidden_state, context = super().forward(
            probe,
            attention_mask=None,
            context=hidden_state,
        )
        hidden_state, _ = self.output_projection(hidden_state)
        hidden_state = hidden_state.transpose(0, 1)
        return hidden_state
