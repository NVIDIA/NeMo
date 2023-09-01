# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The parent decoder class implementation for model_config and tensorrt_llm conversion."""

from abc import ABC, abstractmethod
from typing import Optional

import tensorrt as trt
from transformers.activations import ACT2FN

from ..model_config import (
    QUANTIZATION_NONE,
    AttentionConfig,
    DecoderLayerConfig,
    LayernormConfig,
    MLPConfig,
)
from ..quantization_utils import quantize_linear
from ..tensor_utils import (
    get_tensor_parallel_group,
)


def _get_hidden_act(act_func):
    """Returns the name of the hidden activation functon based on ACT2FN."""
    if isinstance(act_func, str):
        return act_func

    for name, func in ACT2FN.items():
        if isinstance(func, tuple):
            if isinstance(act_func, func[0]):
                return name
        elif isinstance(act_func, func):
            return name
    assert False, f"Cannot find name for {act_func}"


class DecoderLayerConfigBuilder(ABC):
    """A config builder that translate the LLM decoder layer to the DecoderLayerConfig."""

    @abstractmethod
    def hidden_act_fn(self, layer):
        """Returns the hidden act fn in the MLP layer, e.g. SiLUActivation or NewGELUActivation."""
        pass

    @abstractmethod
    def infer_num_attention_heads(self, layer):
        """Returns the num of attention heads of the layer."""
        pass

    @abstractmethod
    def infer_max_position_embeddings(self, layer):
        """Returns the max positional embeddings of the layer."""
        pass

    @abstractmethod
    def build_input_layernorm(self, layer) -> LayernormConfig:
        """Returns the built input layernorm layer."""
        pass

    @abstractmethod
    def build_attention(self, layer) -> AttentionConfig:
        """Returns the built attention layer."""
        pass

    @abstractmethod
    def build_mlp(self, layer) -> MLPConfig:
        """Returns the built mlp layer."""
        pass

    @abstractmethod
    def build_post_layernorm(self, layer) -> Optional[LayernormConfig]:
        """Returns the built post layernorm."""
        pass

    def __init__(
        self,
        decoder_type: str,
        dtype: trt.DataType = trt.float16,
        rank: int = 0,
        tensor_parallel: int = 1,
    ):
        """Initializes the DecoderLayerConfigBuilder."""
        self.decoder_type = decoder_type
        self.dtype = dtype
        self.rank = rank
        self.tensor_parallel = tensor_parallel

    def build_layer(self, layer) -> DecoderLayerConfig:
        """Builds the decoder layer and returns the DecoderLayer."""
        decoder = DecoderLayerConfig()

        decoder.decoder_type = self.decoder_type
        decoder.num_attention_heads = self.infer_num_attention_heads(layer)
        decoder.num_kv_heads = self.infer_num_kv_heads(layer)
        decoder.max_position_embeddings = self.infer_max_position_embeddings(layer)

        decoder.input_layernorm = self.build_input_layernorm(layer)
        decoder.attention = self.build_attention(layer)
        decoder.post_layernorm = self.build_post_layernorm(layer)
        decoder.mlp = self.build_mlp(layer)
        decoder.mlp.hidden_act = _get_hidden_act(self.hidden_act_fn(layer)).split("_")[0]

        return decoder

    def infer_num_kv_heads(self, layer):
        """Returns the num of key value heads of the layer."""
        return self.infer_num_attention_heads(layer)


class DecoderLayerBuilder(ABC):
    """An abstracted transformer decoder layer with tensorrt_llm implementation taking DecoderLayerConfig as the input.

    Individual decoder layers are supposed to extend this class and implement the customized
    abstracted method.
    """

    @abstractmethod
    def build_decoder(self, layer):
        """Returns the built decoder layer."""
        pass

    def __init__(
        self,
        layer: DecoderLayerConfig,
        layer_id: int,
        num_layers: int,
        dtype: trt.DataType = trt.float16,
        quantization: str = QUANTIZATION_NONE,
        rank: int = 0,
        tensor_parallel: int = 1,
    ):
        """Initializes the DecoderLayer."""
        super().__init__()
        assert isinstance(dtype, trt.DataType)
        self.layer_id = layer_id
        self.num_layers = num_layers
        self.dtype = dtype
        self.quantization = quantization
        self.rank = rank
        self.tensor_parallel = tensor_parallel
        self.tp_group = get_tensor_parallel_group(tensor_parallel)

        self.hidden_size = layer.hidden_size
        self.num_attention_heads = layer.num_attention_heads
        self.num_kv_heads = (
            layer.num_kv_heads if layer.num_kv_heads > 0 else layer.num_attention_heads
        )

        assert (
            self.num_attention_heads % self.num_kv_heads
        ) == 0, "MQA/GQA requires the number of heads to be divisible by the number of K/V heads."
        assert (self.num_kv_heads % self.tensor_parallel) == 0 or (
            self.tensor_parallel % self.num_kv_heads
        ) == 0, (
            "MQA/GQA requires either the number of K/V heads to be divisible by the number of GPUs"
            " OR the number of GPUs to be divisible by the number of K/V heads."
        )

        self.max_position_embeddings = layer.max_position_embeddings
        self.hidden_act = layer.mlp.hidden_act

        self.decoder = self.build_decoder(layer)
        self.assign_weights(layer)
        self.quantize(layer)

    def assign_weights(self, layer: DecoderLayerConfig):
        """Assign the weights to the attention tensorrt_llm layer."""
        self.decoder.input_layernorm.weight.value = layer.input_layernorm.weight
        if layer.input_layernorm.bias is not None:
            self.decoder.input_layernorm.bias.value = layer.input_layernorm.bias

        self.decoder.attention.qkv.weight.value = layer.attention.qkv.weight
        if layer.attention.qkv.bias is not None:
            self.decoder.attention.qkv.bias.value = layer.attention.qkv.bias

        self.decoder.attention.dense.weight.value = layer.attention.dense.weight
        if self.decoder.attention.dense.bias is not None:
            self.decoder.attention.dense.bias.value = layer.attention.dense.bias

        if layer.post_layernorm is not None:
            self.decoder.post_layernorm.weight.value = layer.post_layernorm.weight
            if layer.post_layernorm.bias is not None:
                self.decoder.post_layernorm.bias.value = layer.post_layernorm.bias

        self.decoder.mlp.fc.weight.value = layer.mlp.fc.weight
        self.decoder.mlp.proj.weight.value = layer.mlp.proj.weight
        bias = layer.mlp.fc.bias is not None
        if bias:
            self.decoder.mlp.fc.bias.value = layer.mlp.fc.bias
            self.decoder.mlp.proj.bias.value = layer.mlp.proj.bias

        if layer.mlp.gate:
            self.decoder.mlp.gate.weight.value = layer.mlp.gate.weight
            if bias:
                self.decoder.mlp.gate.bias.value = layer.mlp.gate.bias

    def quantize(self, layer: DecoderLayerConfig):
        """Quantizes the decoder layer based on the layer config."""
        self.decoder.attention.qkv = quantize_linear(
            self.decoder.attention.qkv, self.quantization, layer.attention.qkv
        )
        self.decoder.attention.dense = quantize_linear(
            self.decoder.attention.dense, self.quantization, layer.attention.dense
        )
        self.decoder.mlp.fc = quantize_linear(self.decoder.mlp.fc, self.quantization, layer.mlp.fc)
        self.decoder.mlp.proj = quantize_linear(
            self.decoder.mlp.proj, self.quantization, layer.mlp.proj
        )

        if hasattr(self.decoder.mlp, "gate"):
            self.decoder.mlp.gate = quantize_linear(
                self.decoder.mlp.gate, self.quantization, layer.mlp.gate
            )
