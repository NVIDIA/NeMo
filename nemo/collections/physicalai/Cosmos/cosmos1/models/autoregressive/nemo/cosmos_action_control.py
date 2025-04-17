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

# pylint: disable=C0115,C0116,C0301

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
from megatron.core import InferenceParams
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel

from cosmos1.models.autoregressive.nemo.cosmos import CosmosConfig4B, CosmosConfig12B
from cosmos1.models.autoregressive.nemo.cosmos_video2world import CosmosVideo2WorldConfig, CosmosVideo2WorldModel
from cosmos1.models.autoregressive.nemo.inference.inference_controller import CosmosActionControlInferenceWrapper


@dataclass
class CosmosActionControlConfig(CosmosVideo2WorldConfig):
    action_dim: int = 7
    forward_step_fn: Callable = lambda model, batch: model(**batch)
    latent_shape: tuple[int, int, int] = (2, 30, 40)
    pad_to_multiple_of: Optional[int] = 128
    seq_length: int = 2_432

    def configure_model(self, tokenizer) -> "MCoreGPTModel":
        model = super().configure_model(tokenizer)

        model.action_mlp = nn.Sequential(
            nn.Linear(self.action_dim, self.crossattn_emb_size // 4),
            nn.ReLU(),
            nn.Linear(self.crossattn_emb_size // 4, self.crossattn_emb_size),
        )
        return model


class CosmosActionControlModel(CosmosVideo2WorldModel):
    def forward(
        self,
        tokens: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        abs_pos_embed: torch.Tensor,
        action: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        inference_params: InferenceParams | None = None,
    ) -> torch.Tensor:
        # Apply the action MLP to the action to generate the context vector
        mcore_model = self
        while not isinstance(mcore_model, MCoreGPTModel) and hasattr(mcore_model, "module"):
            mcore_model = mcore_model.module

        action_mlp_out = mcore_model.action_mlp(action)
        context = action_mlp_out.unsqueeze(0).repeat(self.config.action_dim, 1, 1)

        return super().forward(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            extra_block_kwargs={
                "context": context,
                "extra_positional_embeddings": abs_pos_embed,
            },
            packed_seq_params=None,
            inference_params=inference_params,
        )

    def get_inference_wrapper(self, params_dtype, inference_batch_times_seqlen_threshold) -> torch.Tensor:
        # This is to get the MCore model required in GPTInferenceWrapper.
        mcore_model = self.module
        vocab_size = self.config.vocab_size

        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=mcore_model.config.hidden_size,
            params_dtype=params_dtype,
            inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
            padded_vocab_size=vocab_size,
        )

        model_inference_wrapper = CosmosActionControlInferenceWrapper(
            mcore_model, inference_wrapper_config, self.config
        )
        return model_inference_wrapper


@dataclass
class CosmosConfigActionControl5B(CosmosActionControlConfig, CosmosConfig4B):
    make_vocab_size_divisible_by: int = 64


@dataclass
class CosmosConfigActionControl13B(CosmosActionControlConfig, CosmosConfig12B):
    make_vocab_size_divisible_by: int = 128
