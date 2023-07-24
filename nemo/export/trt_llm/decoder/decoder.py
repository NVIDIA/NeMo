# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from abc import ABC, abstractmethod

import tensorrt as trt
from tensorrt_llm.functional import RaggedTensor
from tensorrt_llm.module import Module

from ..tensorrt_llm_utils import (
    get_hidden_act,
    get_tensor_parallel_group,
)


class DecoderLayer(Module, ABC):
    """An abstracted transformer decoder layer with tensorrt_llm implementation.

    Individual decoder layers are supposed to extend this class and implement the customized
    abstracted method.
    """

    @abstractmethod
    def hidden_act_fn(self, layer):
        """Returns the hidden act fn in the MLP layer, e.g. SiLUActivation or NewGELUActivation"""
        pass

    @abstractmethod
    def infer_hidden_size(self, layer):
        """Returns the hidden size of the layer."""
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
    def build_input_layernorm(self, layer):
        """Returns the built input layernorm layer."""
        pass

    @abstractmethod
    def build_attention(self, layer):
        """Returns the built attention layer."""
        pass

    @abstractmethod
    def build_mlp(self, layer):
        """Returns the built mlp layer."""
        pass

    @abstractmethod
    def build_post_layernorm(self, layer):
        """Returns the built post layernorm"""
        pass

    @abstractmethod
    def post_attention_forward(self, residual, hidden_states, attention_output):
        """Returns an updated hidden_states post attention layer forward."""
        pass

    def __init__(self, layer, num_layers, dtype=trt.float16, rank=0, tensor_parallel=1):
        super().__init__()
        assert isinstance(dtype, trt.DataType)
        self.num_layers = num_layers
        self.dtype = dtype
        self.rank = rank
        self.tensor_parallel = tensor_parallel
        self.tp_group = get_tensor_parallel_group(tensor_parallel)

        self.hidden_size = self.infer_hidden_size(layer)
        self.num_attention_heads = self.infer_num_attention_heads(layer)
        self.max_position_embeddings = self.infer_max_position_embeddings(layer)
        self.hidden_act = get_hidden_act(self.hidden_act_fn(layer)).split("_")[0]

        self.input_layernorm = self.build_input_layernorm(layer)
        self.attention = self.build_attention(layer)
        self.post_layernorm = self.build_post_layernorm(layer)
        self.mlp = self.build_mlp(layer)

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
        )

        if use_cache:
            attention_output, presents = attention_output

        hidden_states = self.post_attention_forward(residual, hidden_states, attention_output.data)

        hidden_states = RaggedTensor.from_row_lengths(
            hidden_states, input_lengths, max_input_length
        )

        if use_cache:
            return (hidden_states, presents)
        return hidden_states
