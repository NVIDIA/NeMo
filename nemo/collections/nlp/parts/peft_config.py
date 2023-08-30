# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from typing import Dict
from omegaconf import DictConfig
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    LoraKQVAdapterConfig,
    MLPInfusedAdapterConfig,
    InfusedAdapterConfig,
    PromptEncoderAdapterConfig,
    ParallelLinearAdapterConfig,
)

class PEFTConfig:
    # superclass for adapter name and config
    def __init__(self, cfg: DictConfig, peft_cfg: DictConfig, name_key_to_cfg: Dict):
        self.name_key_to_cfg = name_key_to_cfg

        self.layer_selection = peft_cfg.get("layer_selection", None)
        if self.layer_selection is None:
            self.layer_selection = list(range(1, cfg.num_layers + 1))

    def get_config_dict(self):
        return self.name_key_to_cfg

class LoraPEFTConfig(PEFTConfig):
    def __init__(self, cfg):
        lora_cfg = cfg.peft.lora_tuning
        if cfg.get("kv_channels", None) is None:
            assert (
                cfg.hidden_size % cfg.num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = cfg.hidden_size // cfg.num_attention_heads
        else:
            kv_channels = cfg.kv_channels
        projection_size = kv_channels * cfg.num_attention_heads

        adapter_cfg = LoraKQVAdapterConfig(
            in_features=cfg.hidden_size,
            out_features=3 * projection_size,
            dim=lora_cfg.adapter_dim,
            norm_position=None,
            norm_type=None,
            activation="identity",
            column_init_method=lora_cfg.get("column_init_method", "normal"),
            row_init_method=lora_cfg.get("row_init_method", "zero"),
            gather_output=False,
            dropout=lora_cfg.adapter_dropout,
        )

        name_key_to_cfg = {
            AdapterName.LORA_KQV_ADAPTER: adapter_cfg,
        }

        super().__init__(cfg, lora_cfg, name_key_to_cfg)


class IA3PEFTConfig(PEFTConfig):
    def __init__(self, cfg):
        mlp_infused_adapter_cfg = MLPInfusedAdapterConfig(
            in_features=cfg.ffn_hidden_size // cfg.tensor_model_parallel_size
        )
        infused_adapter_cfg = InfusedAdapterConfig(in_features=cfg.hidden_size // cfg.tensor_model_parallel_size)

        name_key_to_cfg = {
            AdapterName.KEY_INFUSED: infused_adapter_cfg,
            AdapterName.VALUE_INFUSED: infused_adapter_cfg,
            AdapterName.MLP_INFUSED: mlp_infused_adapter_cfg,
        }

        super().__init__(cfg, cfg.peft.ia3_tuning, name_key_to_cfg)


class PtuningPEFTConfig(PEFTConfig):
    def __init__(self, cfg):
        adapter_cfg = PromptEncoderAdapterConfig(
            cfg.peft.p_tuning.virtual_tokens,
            cfg.peft.p_tuning.bottleneck_dim,
            cfg.peft.p_tuning.embedding_dim,
            cfg.peft.p_tuning.init_std,
            cfg.hidden_size,
        )
        name_key_to_cfg = {AdapterName.PTUNING_ADAPTER: adapter_cfg}

        super().__init__(cfg, cfg.peft.p_tuning, name_key_to_cfg)


class AdapterPEFTConfig(PEFTConfig):
    def __init__(self, cfg):
        adapter_tuning_cfg = cfg.peft.adapter_tuning
        adapter_cfg = ParallelLinearAdapterConfig(
            in_features=cfg.hidden_size,
            out_features=cfg.hidden_size,
            dim=adapter_tuning_cfg.adapter_dim,
            norm_position=adapter_tuning_cfg.get("norm_position", "pre"),
            norm_type=adapter_tuning_cfg.get("norm_type", "mixedfusedlayernorm"),
            column_init_method=adapter_tuning_cfg.get("column_init_method", "xavier"),
            row_init_method=adapter_tuning_cfg.get("row_init_method", "zero"),
            dropout=adapter_tuning_cfg.adapter_dropout,
        )
        name_key_to_cfg = {
            AdapterName.PRE_ATTN_ADAPTER: adapter_cfg,
            AdapterName.POST_ATTN_ADAPTER: adapter_cfg,
        }

        super().__init__(cfg, adapter_tuning_cfg, name_key_to_cfg)


PEFT_CONFIG_MAP = {
    "adapter": AdapterPEFTConfig,
    "ia3": IA3PEFTConfig,
    "ptuning": PtuningPEFTConfig,
    "lora": LoraPEFTConfig,
}
