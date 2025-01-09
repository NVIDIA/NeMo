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
from megatron.core.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import init_method_normal


class ImageOutputProjectionModule(MegatronModule):
    def __init__(
        self,
        config,
        encoder_config,
        transformer_encoder_layer_spec,
        transformer_decoder_layer_spec,
        output_linear_projection,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.encoder_config: TransformerConfig = encoder_config
        self.transformer_encoder_layer_spec: ModuleSpec = transformer_encoder_layer_spec
        self.transformer_decoder_layer_spec: ModuleSpec = transformer_decoder_layer_spec
        self.model_type = ModelType.encoder_and_decoder
        self.num_query_token = self.config.num_query_token

        # transformer encoder layer with no bidirectional mask
        self.encoder = TransformerBlock(
            config=self.encoder_config,
            spec=self.transformer_encoder_layer_spec,
        )

        # transformer decoder layer with self and cross attention
        self.decoder = TransformerBlock(
            config=self.config,
            spec=self.transformer_decoder_layer_spec,
        )

        # learnable query embeddings input to transformer decoder
        self.probe = torch.nn.Parameter(torch.randn(self.num_query_token, 1, config.hidden_size))
        # oup linear layer
        self.output_projection = build_module(
            output_linear_projection,
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
        probe = self.probe.repeat(1, batch_size, 1)
        hidden_state = hidden_state.transpose(0, 1)
        encoder_hidden_states = self.encoder(hidden_state, attention_mask=None)
        decoder_hidden_states = self.decoder(probe, attention_mask=None, context=encoder_hidden_states)
        output_projection, _ = self.output_projection(decoder_hidden_states)
        output_projection = output_projection.transpose(0, 1)
        return output_projection
