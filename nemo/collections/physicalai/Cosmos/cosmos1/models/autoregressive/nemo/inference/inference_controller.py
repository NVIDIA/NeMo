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

from typing import Any

import torch
from cosmos1.models.autoregressive.modules.embedding import SinCosPosEmbAxisTE
from megatron.core import tensor_parallel
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from megatron.core.inference.text_generation_controllers.simple_text_generation_controller import (
    SimpleTextGenerationController,
)
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel


class CosmosInferenceWrapper(GPTInferenceWrapper):
    def __init__(
        self, model: GPTModel, inference_wrapper_config: InferenceWrapperConfig, config  #: CosmosVideo2WorldConfig
    ):
        super().__init__(model, inference_wrapper_config)
        self.config = config

    def prep_model_for_inference(self, prompts_tokens: torch.Tensor):
        super().prep_model_for_inference(prompts_tokens=prompts_tokens)
        self.abs_pos_emb = self._initialize_abs_pos_emb()

    def _initialize_abs_pos_emb(self):
        pos_emb = SinCosPosEmbAxisTE(
            dim=self.config.hidden_size,
            latent_shape=self.config.latent_shape,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
        )
        abs_pos_emb = pos_emb.forward(training_type=self.config.training_type)
        abs_pos_emb = abs_pos_emb.transpose(0, 1).contiguous()
        return abs_pos_emb

    def get_batch_for_context_window(
        self, inference_input, context_start_position: int, context_end_position: int
    ) -> dict[str, Any]:
        data_at_step_idx = super().get_batch_for_context_window(
            inference_input, context_start_position, context_end_position
        )
        absposembed2use = self.abs_pos_emb[context_start_position:context_end_position, :, :]
        data_at_step_idx["extra_positional_embeddings"] = absposembed2use
        return data_at_step_idx

    def set_context_tokens(self, batch_context_tokens):
        self.context_tokens = batch_context_tokens

    def forward_pass_without_pipeline_parallel(self, inference_input: dict[str, Any]) -> torch.Tensor:
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        attention_mask = inference_input["attention_mask"]
        abs_pos_embed = inference_input["extra_positional_embeddings"]

        assert hasattr(
            self, "context_tokens"
        ), "Expected to have context tokens. Not present. Call set_context_tokens with the encoder embeddings"
        extra_block_kwargs = {"context": self.context_tokens, "extra_positional_embeddings": abs_pos_embed}
        packed_seq_params = None

        logits = self.model(
            tokens,
            position_ids,
            attention_mask,
            inference_params=self.inference_params,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
        )
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
        self.inference_params.sequence_len_offset += tokens.size(1)

        return logits


class CosmosTextGenerationController(SimpleTextGenerationController):
    def generate_all_output_tokens_static_batch(self, active_requests, active_streams=None):
        batch_context_tokens = (
            list(map(lambda request: request.encoder_prompt, active_requests.values()))[0]
            .to(torch.bfloat16)
            .permute(1, 0, 2)
        )
        self.inference_wrapped_model.set_context_tokens(batch_context_tokens)
        active_requests = super().generate_all_output_tokens_static_batch(active_requests, active_streams)
        return active_requests


class CosmosActionControlInferenceWrapper(CosmosInferenceWrapper):
    def forward_pass_without_pipeline_parallel(self, inference_input: dict[str, Any]) -> torch.Tensor:
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        attention_mask = inference_input["attention_mask"]
        abs_pos_embed = inference_input["extra_positional_embeddings"]

        assert hasattr(
            self, "context_tokens"
        ), "Expected to have context tokens. Not present. Call set_context_tokens with the encoder embeddings"
        mcore_model = self.model
        while not isinstance(mcore_model, MCoreGPTModel) and hasattr(mcore_model, "module"):
            mcore_model = mcore_model.module
        action_mlp_out = mcore_model.action_mlp(self.context_tokens.cuda())
        context = action_mlp_out.unsqueeze(0).repeat(self.config.action_dim, 1, 1)

        extra_block_kwargs = {"extra_positional_embeddings": abs_pos_embed.cuda(), "context": context}
        packed_seq_params = None
        logits = self.model(
            tokens.cuda(),
            position_ids.cuda(),
            attention_mask.cuda(),
            inference_params=self.inference_params,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
        )

        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
        self.inference_params.sequence_len_offset += tokens.size(1)

        return logits


class CosmosActionGenerationController(SimpleTextGenerationController):
    def generate_all_output_tokens_static_batch(self, active_requests, active_streams=None):
        batch_actions = list(map(lambda request: request.encoder_prompt, active_requests.values()))[0].to(
            torch.bfloat16
        )
        self.inference_wrapped_model.set_context_tokens(batch_actions)
        active_requests = super().generate_all_output_tokens_static_batch(active_requests, active_streams)
        return active_requests
