# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from nemo.collections.nlp.models.language_modeling.megatron.griffin.griffin_layer_spec import (
    griffin_recurrent_layer_with_transformer_engine_spec, 
    griffin_mqa_layer_with_transformer_engine_spec
)
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from torch import nn

def get_griffin_layers(num_layers):
    dict_spec = {"Recurrent_Layer": griffin_recurrent_layer_with_transformer_engine_spec,
                "Attention_Layer": griffin_mqa_layer_with_transformer_engine_spec}

    griffin_layers = []
    for i in range(num_layers):
        if i % 3 == 2:
            griffin_layers.append(dict_spec["Attention_Layer"])
        else:
            griffin_layers.append(dict_spec["Recurrent_Layer"])

    return griffin_layers
def create_block(
    config,
    layer_spec,
    layer_idx=None,
):
    block = build_module(
        layer_spec,    
        config,
    )
    block.layer_idx = layer_idx
    block.layer_number = layer_idx
    return block

class GriffinStack(LanguageModule):
    def __init__(
        self,
        config: TransformerConfig,
    ):

        super().__init__(config)
        self.config = config
        self.griffin_layers = get_griffin_layers(self.config.num_layers)

        self.layers = nn.ModuleList(
            [
                create_block(
                    self.config,
                    layer_spec,
                    layer_idx=i,
                )
                for i, layer_spec in enumerate(self.griffin_layers)
            ]
        )
        self.final_layernorm = TENorm(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )

    def forward(
            self,
             hidden_states,
            attention_mask,
            rotary_pos_emb
    ):

        for layer in self.layers:

            hidden_states, _ = layer(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb
            )

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states

