import torch

from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    LoraKQVAdapterConfig,
    PromptEncoderAdapterConfig
)
from nemo.core import adapter_mixins
from nemo.utils import model_utils

from megatron.core.models.gpt.gpt_embedding import GPTEmbedding
from megatron.core.transformer.custom_layers.transformer_engine import TELinear


def swap_mcore_mixin(module, mcore_mixin):
    """
    Casts module to mcore_mixin and register corresponding adapters.
    """
    module.__class__ = mcore_mixin
    module.mcore_register_adapters()


class McoreAdapterModuleMixin(adapter_mixins.AdapterModuleMixin):
    def mcore_register_adapters(self):
        raise NotImplementedError("Mcore mixins should implement setup_adapters on a subclass of MyBase")


class LoraTELinear(TELinear, McoreAdapterModuleMixin):
    def mcore_register_adapters(self):
        self.set_accepted_adapter_types(
            [
                LoraKQVAdapterConfig._target_,  # only self attn (packed qkv) for now
                # LoraQAdapterConfig._target_,
                # LoraKVAdapterConfig._target_,
            ]
        )

    def forward(self, x):
        mixed_x_layer, bias = super().forward(x)

        if self.is_adapter_available():
                lora_kqv_adapter = self.get_adapter_module(AdapterName.LORA_KQV_ADAPTER)
                if lora_kqv_adapter:
                    lora_mixed_x_layer = lora_kqv_adapter(x)
                    mixed_x_layer = mixed_x_layer + lora_mixed_x_layer
        
        return mixed_x_layer, bias


class PtuningGPTEmbedding(GPTEmbedding, McoreAdapterModuleMixin):
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