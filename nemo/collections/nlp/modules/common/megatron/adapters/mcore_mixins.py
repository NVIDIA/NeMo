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
from megatron.core.models.gpt.gpt_embedding import GPTEmbedding
from megatron.core.transformer.attention import SelfAttention

from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    LoraKQVAdapterConfig,
    PromptEncoderAdapterConfig,
)
from nemo.core import adapter_mixins


def swap_mcore_mixin(module, mcore_mixin):
    """
    Casts module to mcore_mixin and register corresponding adapters.
    """
    module.__class__ = mcore_mixin
    module.mcore_register_adapters()


class MCoreAdapterModuleMixin(adapter_mixins.AdapterModuleMixin):
    def mcore_register_adapters(self):
        raise NotImplementedError("Mcore mixins should implement setup_adapters on a subclass of MyBase")


class MCoreSelfAttentionMixin(SelfAttention, MCoreAdapterModuleMixin):
    def mcore_register_adapters(self):
        self.set_accepted_adapter_types([LoraKQVAdapterConfig._target_])  # only self attn (packed qkv) for now

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # LoRA logic
        if self.is_adapter_available():
            lora_kqv_adapter = self.get_adapter_module(AdapterName.LORA_KQV_ADAPTER)
            if lora_kqv_adapter:
                lora_mixed_qkv = lora_kqv_adapter(hidden_states)
                mixed_qkv = mixed_qkv + lora_mixed_qkv

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        (query, key, value) = torch.split(
            mixed_qkv,
            [
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    * self.hidden_size_per_attention_head
                ),
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ],
            dim=3,
        )
        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        return query, key, value


class MCoreGPTEmbeddingMixin(GPTEmbedding, MCoreAdapterModuleMixin):
    def mcore_register_adapters(self):
        self.set_accepted_adapter_types([PromptEncoderAdapterConfig._target_])

    def forward(self, input_ids, position_ids):
        encoder_input = super().forward(input_ids, position_ids)

        if self.is_adapter_available():
            _sq, _bs, _hs = encoder_input.size()
            ptuning_adapter = self.get_adapter_module(AdapterName.PTUNING_ADAPTER)
            v = ptuning_adapter.virtual_tokens
            if ptuning_adapter and _sq >= v:  # The sequence should be longer the v to insert virtual embeddings.
                virtual_embeddings = ptuning_adapter(_bs)
                encoder_input = encoder_input[
                    v:, :, :
                ]  # the first v tokens are pads so that they can be swapped out with virtual embeddings.
                encoder_input = torch.concat([virtual_embeddings, encoder_input], dim=0)
        return encoder_input
