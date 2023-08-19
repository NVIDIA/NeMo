# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from abc import ABC, abstractmethod
from typing import Optional

import tensorrt as trt
from tensorrt_llm.functional import RaggedTensor, non_gated_version
from tensorrt_llm.layers import MLP, GatedMLP
from tensorrt_llm.module import Module
from transformers.activations import ACT2FN

from ..model_config import QUANTIZATION_NONE, AttentionConfig, DecoderLayerConfig, LayernormConfig, MLPConfig
from ..quantization_utils import quantize_linear
from ..tensor_utils import get_tensor_parallel_group
from ..tensorrt_llm_utils import build_layernorm_from_config


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
    """A config builder that translate the LLM decoder layer to the DecoderLayerConfig"""

    @abstractmethod
    def hidden_act_fn(self, layer):
        """Returns the hidden act fn in the MLP layer, e.g. SiLUActivation or NewGELUActivation"""
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
        """Returns the built post layernorm"""
        pass

    def __init__(
        self,
        decoder_type: str,
        dtype: trt.DataType = trt.float16,
        rank: int = 0,
        tensor_parallel: int = 1,
    ):
        self.decoder_type = decoder_type
        self.dtype = dtype
        self.rank = rank
        self.tensor_parallel = tensor_parallel

    def build_layer(self, layer) -> DecoderLayerConfig:
        decoder = DecoderLayerConfig()

        decoder.decoder_type = self.decoder_type
        decoder.num_attention_heads = self.infer_num_attention_heads(layer)
        decoder.max_position_embeddings = self.infer_max_position_embeddings(layer)

        decoder.input_layernorm = self.build_input_layernorm(layer)
        decoder.attention = self.build_attention(layer)
        decoder.post_layernorm = self.build_post_layernorm(layer)
        decoder.mlp = self.build_mlp(layer)
        decoder.mlp.hidden_act = _get_hidden_act(self.hidden_act_fn(layer)).split("_")[0]

        return decoder


class DecoderLayer(Module, ABC):
    """An abstracted transformer decoder layer with tensorrt_llm implementation taking DecoderLayerConfig as the input.

    Individual decoder layers are supposed to extend this class and implement the customized
    abstracted method.
    """

    @abstractmethod
    def build_attention(self, layer):
        """Returns the built attention layer."""
        pass

    @abstractmethod
    def post_attention_forward(self, residual, hidden_states, attention_output):
        """Returns an updated hidden_states post attention layer forward."""
        pass

    def __init__(
        self,
        layer: DecoderLayerConfig,
        num_layers: int,
        dtype: trt.DataType = trt.float16,
        quantization: str = QUANTIZATION_NONE,
        rank: int = 0,
        tensor_parallel: int = 1,
    ):
        super().__init__()
        assert isinstance(dtype, trt.DataType)
        self.num_layers = num_layers
        self.dtype = dtype
        self.quantization = quantization
        self.rank = rank
        self.tensor_parallel = tensor_parallel
        self.tp_group = get_tensor_parallel_group(tensor_parallel)

        self.hidden_size = layer.hidden_size
        self.num_attention_heads = layer.num_attention_heads
        self.max_position_embeddings = layer.max_position_embeddings
        self.hidden_act = layer.mlp.hidden_act

        self.input_layernorm = build_layernorm_from_config(layer.input_layernorm, self.dtype)
        self.attention = self.build_attention(layer)
        self.assign_attention_weights(layer)
        self.post_layernorm = build_layernorm_from_config(layer.post_layernorm, self.dtype)
        self.build_mlp(layer)
        self.quantize(layer)

    def assign_attention_weights(self, layer: DecoderLayerConfig):
        self.attention.qkv.weight.value = layer.attention.qkv.weight
        if layer.attention.qkv.bias is not None:
            self.attention.qkv.bias.value = layer.attention.qkv.bias

        self.attention.dense.weight.value = layer.attention.dense.weight
        if layer.attention.dense.bias is not None:
            self.attention.dense.bias.value = layer.attention.dense.bias

    def build_mlp(self, layer: DecoderLayerConfig):
        """Helper function to build the MLP layer from the DecoderLayerConfig."""
        mlp_builder = GatedMLP if layer.mlp.gate else MLP
        bias = layer.mlp.fc.bias is not None
        self.mlp = mlp_builder(
            layer.hidden_size,
            layer.ffn_hidden_size_local * self.tensor_parallel,
            non_gated_version(self.hidden_act),
            bias,
            self.dtype,
            self.tp_group,
            self.tensor_parallel,
        )

        self.mlp.fc.weight.value = layer.mlp.fc.weight
        self.mlp.proj.weight.value = layer.mlp.proj.weight

        if bias:
            self.mlp.fc.bias.value = layer.mlp.fc.bias
            self.mlp.proj.bias.value = layer.mlp.proj.bias

        if layer.mlp.gate:
            self.mlp.gate.weight.value = layer.mlp.gate.weight
            if bias:
                self.mlp.gate.bias.value = layer.mlp.gate.bias

    def quantize(self, layer: DecoderLayerConfig):
        """Quantizes the decoder layer based on the layer config."""
        self.attention.qkv = quantize_linear(self.attention.qkv, self.quantization, layer.attention.qkv)
        self.attention.dense = quantize_linear(self.attention.dense, self.quantization, layer.attention.dense)
        self.mlp.fc = quantize_linear(self.mlp.fc, self.quantization, layer.mlp.fc)
        self.mlp.proj = quantize_linear(self.mlp.proj, self.quantization, layer.mlp.proj)

        if hasattr(self.mlp, "gate"):
            self.mlp.gate = quantize_linear(self.mlp.gate, self.quantization, layer.mlp.gate)

    def forward(
        self,
        hidden_states: RaggedTensor,
        attention_mask=None,
        past_key_value=None,
        sequence_length=None,
        past_key_value_length=None,
        masked_tokens=None,
        use_cache=False,
        cache_indirection=None,
        kv_cache_block_pointers=None,
        inflight_batching_args=None,
        past_key_value_pointers=None,
    ):
        """Forward function for the decoder layer."""

        assert isinstance(hidden_states, RaggedTensor)
        # unpack the RaggedTensor since some layers like MLP, LayerNorm only need data tensor
        input_lengths = hidden_states.row_lengths
        max_input_length = hidden_states.max_row_length
        hidden_states = hidden_states.data

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(
            RaggedTensor.from_row_lengths(hidden_states, input_lengths, max_input_length),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            sequence_length=sequence_length,
            past_key_value_length=past_key_value_length,
            masked_tokens=masked_tokens,
            use_cache=use_cache,
            cache_indirection=cache_indirection,
            # Disable the additional features not compatible with GPTJ
            # TODO: enable them in the future.
            # kv_cache_block_pointers=kv_cache_block_pointers,
            # inflight_batching_args=inflight_batching_args,
            # past_key_value_pointers=past_key_value_pointers,
        )

        if use_cache:
            attention_output, presents = attention_output

        hidden_states = self.post_attention_forward(residual, hidden_states, attention_output.data)

        hidden_states = RaggedTensor.from_row_lengths(hidden_states, input_lengths, max_input_length)

        if use_cache:
            return (hidden_states, presents)
        return hidden_states
