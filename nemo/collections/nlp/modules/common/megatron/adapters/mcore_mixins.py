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
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TELayerNormColumnParallelLinear,
)
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_layer import TransformerLayer

from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    InfusedAdapterConfig,
    Lora4HtoHAdapterConfig,
    LoraDenseAttentionAdapterConfig,
    LoraHto4HAdapterConfig,
    LoraKQVAdapterConfig,
    MLPInfusedAdapterConfig,
    ParallelLinearAdapterConfig,
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
        """
        Performs any necessary setup after swapping class.
        Must use self.set_accepted_adapter_types([<NeMo adapter config>_target_]) to register adapter.
        """
        raise NotImplementedError("Mcore mixins should implement setup_adapters on a subclass of MyBase")


class MCoreSelfAttentionMixin(SelfAttention, MCoreAdapterModuleMixin):
    def mcore_register_adapters(self):
        """
        Setup NeMo LoRA or IA3 adapter to this MCore layer.
        """
        self.set_accepted_adapter_types(
            [LoraKQVAdapterConfig._target_, LoraDenseAttentionAdapterConfig._target_, InfusedAdapterConfig._target_]
        )
        self.linear_qkv.return_layernorm_output = True  # need layernorm output for lora mlp

    def _lora_attn_qkv_adapter_hook(self, hidden_states, linear_qkv_output):
        layernorm_output = None

        # In megatron/core/models/gpt/gpt_layer_specs.py TELayerNormColumnParallelLinear is used for linear_qkv.
        # TELayerNormColumnParallelLinear fused LN and linear, both will be returned.
        # In nemo/collections/nlp/models/language_modeling/megatron/falcon/falcon_spec.py TEColumnParallelLinear is used for linear_qkv,
        # which only returns linear.
        if isinstance(self.linear_qkv, TELayerNormColumnParallelLinear):
            mixed_qkv, layernorm_output = linear_qkv_output
        elif isinstance(self.linear_qkv, TEColumnParallelLinear):  # only mixed_qkv
            mixed_qkv = linear_qkv_output
        else:
            raise ValueError(
                f"Unrecognized module type '{type(self.linear_qkv)}' when getting query, key, value tensors for mcore mixins. "
            )

        if self.is_adapter_available():
            lora_kqv_adapter = self.get_adapter_module(AdapterName.LORA_KQV_ADAPTER)
            if lora_kqv_adapter and self.adapter_cfg[AdapterName.LORA_KQV_ADAPTER]['enabled']:
                if isinstance(self.linear_qkv, TELayerNormColumnParallelLinear):
                    lora_mixed_qkv = lora_kqv_adapter(layernorm_output)
                elif isinstance(self.linear_qkv, TEColumnParallelLinear):
                    lora_mixed_qkv = lora_kqv_adapter(hidden_states)
                else:
                    raise ValueError(f"Unrecognized module type '{type(self.linear_qkv)}' when applying lora.")
                mixed_qkv = mixed_qkv + lora_mixed_qkv

        return mixed_qkv

    def _ia3_kv_infused_adapter_hook(self, query, key, value):
        if self.is_adapter_available():
            key_infused_adapter = self.get_adapter_module(AdapterName.KEY_INFUSED)
            value_infused_adapter = self.get_adapter_module(AdapterName.VALUE_INFUSED)
            if key_infused_adapter and self.adapter_cfg[AdapterName.KEY_INFUSED]['enabled']:
                assert value_infused_adapter is not None, "Expected value_infused_adapter not found!"
                kls = key.shape
                key = key_infused_adapter(key.reshape(kls[0], kls[1], -1)).reshape(kls).to(query.dtype)
            if value_infused_adapter and self.adapter_cfg[AdapterName.VALUE_INFUSED]['enabled']:
                assert key_infused_adapter is not None, "Expected key_infused_adapter not found!"
                vls = value.shape
                value = value_infused_adapter(value.reshape(vls[0], vls[1], -1)).reshape(vls).to(query.dtype)
        return key, value

    def _lora_attn_dense_adapter_hook(self, core_attn_out, output):
        if self.is_adapter_available():
            lora_linear_proj_adapter = self.get_adapter_module(AdapterName.LORA_DENSE_ATTENTION_ADAPTER)
            if lora_linear_proj_adapter and self.adapter_cfg[AdapterName.LORA_DENSE_ATTENTION_ADAPTER]['enabled']:
                lora_output = lora_linear_proj_adapter(core_attn_out)
                output = output + lora_output
        return output


class MCoreMLPMixin(MLP, MCoreAdapterModuleMixin):
    def mcore_register_adapters(self):
        """
        Setup NeMo LoRA or IA3 adapter to this MCore layer.
        """
        self.set_accepted_adapter_types(
            [LoraHto4HAdapterConfig._target_, Lora4HtoHAdapterConfig._target_, MLPInfusedAdapterConfig._target_]
        )

    def _lora_mlp_fc1_adapter_hook(self, hidden_states, intermediate_parallel):
        if self.is_adapter_available():
            lora_linear_fc1_adapter = self.get_adapter_module(AdapterName.LORA_Hto4H_ADAPTER)
            if lora_linear_fc1_adapter and self.adapter_cfg[AdapterName.LORA_Hto4H_ADAPTER]['enabled']:
                lora_output = lora_linear_fc1_adapter(hidden_states)
                intermediate_parallel = intermediate_parallel + lora_output
        return intermediate_parallel

    def _lora_mlp_fc2_adapter_hook(self, intermediate_parallel, output):
        if self.is_adapter_available():
            lora_linear_fc2_adapter = self.get_adapter_module(AdapterName.LORA_4HtoH_ADAPTER)
            if lora_linear_fc2_adapter and self.adapter_cfg[AdapterName.LORA_4HtoH_ADAPTER]['enabled']:
                lora_output = lora_linear_fc2_adapter(intermediate_parallel)
                output = output + lora_output
        return output

    def _ia3_mlp_infused_adapter_hook(self, intermediate_parallel):
        if self.is_adapter_available():
            infused_adapter = self.get_adapter_module(AdapterName.MLP_INFUSED)
            if infused_adapter and self.adapter_cfg[AdapterName.MLP_INFUSED]['enabled']:
                intermediate_parallel = infused_adapter(intermediate_parallel)
        return intermediate_parallel


class MCoreGPTEmbeddingMixin(LanguageModelEmbedding, MCoreAdapterModuleMixin):
    def mcore_register_adapters(self):
        """
        Setup NeMo ptuning adapter to this MCore layer.
        """
        self.set_accepted_adapter_types([PromptEncoderAdapterConfig._target_])

    def forward(self, input_ids, position_ids):
        encoder_input = super().forward(input_ids, position_ids)

        if self.is_adapter_available():
            _sq, _bs, _hs = encoder_input.size()
            ptuning_adapter = self.get_adapter_module(AdapterName.PTUNING_ADAPTER)
            v = ptuning_adapter.virtual_tokens
            if (
                ptuning_adapter and self.adapter_cfg[AdapterName.PTUNING_ADAPTER]['enabled'] and _sq >= v
            ):  # The sequence should be longer the v to insert virtual embeddings.
                virtual_embeddings = ptuning_adapter(_bs)
                encoder_input = encoder_input[
                    v:, :, :
                ]  # the first v tokens are pads so that they can be swapped out with virtual embeddings.
                encoder_input = torch.concat([virtual_embeddings, encoder_input], dim=0)
        return encoder_input


class MCoreTransformerLayerMixin(TransformerLayer, MCoreAdapterModuleMixin):
    def mcore_register_adapters(self):
        """
        Setup NeMo (canonical) Adapter to this MCore layer.
        """
        self.set_accepted_adapter_types([ParallelLinearAdapterConfig._target_])

    def _adapter_pre_attn_adapter_hook(self, attention_output_with_bias):
        if self.is_adapter_available():
            adapter_1 = self.get_adapter_module(AdapterName.PRE_ATTN_ADAPTER)
            if adapter_1 and self.adapter_cfg[AdapterName.PRE_ATTN_ADAPTER]['enabled']:
                attention_output, bias = attention_output_with_bias
                attention_output = (
                    adapter_1(attention_output) + attention_output
                )  # simple adapter call with residual connection
                attention_output_with_bias = (attention_output, bias)
        return attention_output_with_bias

    def _adapter_post_attn_adapter_hook(self, mlp_output_with_bias):
        if self.is_adapter_available():
            adapter_2 = self.get_adapter_module(AdapterName.POST_ATTN_ADAPTER)
            if adapter_2 and self.adapter_cfg[AdapterName.POST_ATTN_ADAPTER]['enabled']:
                mlp_output, bias = mlp_output_with_bias
                mlp_output = adapter_2(mlp_output) + mlp_output  # simple adapter call with residual connection
                mlp_output_with_bias = (mlp_output, bias)
        return mlp_output_with_bias
