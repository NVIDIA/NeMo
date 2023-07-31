# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Referrence impl https://gitlab-master.nvidia.com/ftp/tekit/-/blob/main/tensorrt_llm/models/gpt/model.py"""


from collections import OrderedDict
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from tensorrt_llm._common import default_net
from tensorrt_llm._utils import pad_vocab_size
from tensorrt_llm.functional import RaggedTensor, Tensor, assertion, expand_mask, gather_last_token_logits, shape
from tensorrt_llm.layers import ColumnLinear, Embedding, LayerNorm
from tensorrt_llm.module import Module, ModuleList
from torch import nn
from transformers import GPT2Config

from .decoder import build_decoder_layer
from .decoder.nemo import NemoLayer
from .tensorrt_llm_build import build
from .tensorrt_llm_utils import (
    build_embedding,
    build_layernorm,
    get_hidden_size,
    get_tensor_from_file,
    get_tensor_parallel_group,
    split,
    torch_to_np,
)


class ModelBuilder(Module):
    """A generic tensorrt_llm transformer model builder.

    We try to make this module builder as flexibile as possible to cover all transformer conversion usecases.
    """

    def __init__(self, dtype: trt.DataType = trt.float16, rank: int = 0, tensor_parallel: int = 1):
        super().__init__()
        self.max_position_embeddings = 0
        self.dtype = dtype  # trt.float16 or trt.float32
        self.kv_dtype = dtype
        self.rank = rank
        self.tensor_parallel = tensor_parallel
        self.multi_query_mode = False
        assert isinstance(dtype, trt.DataType)

    def add_vocab_embedding(self, embedding: nn.Module):
        """Adds the vocab embedding layer to the model."""
        if type(embedding) == nn.Embedding:
            # Handles the normal embedding path
            self.vocab_size = embedding.num_embeddings
            self.hidden_size = get_hidden_size(embedding)
            self.vocab_embedding = build_embedding(embedding, dtype=self.dtype)
        else:
            raise NotImplementedError(f"{embedding} not supported")

    def add_positional_embedding(self, embedding: nn.Module):
        """Adds the positional embedding layer to the model.

        This API can be optional if the positional embedding layer does not exist in the original model.
        """
        if type(embedding) == nn.Embedding:
            if self.max_position_embeddings != 0:
                assert self.max_position_embeddings == embedding.num_embeddings
            else:
                self.max_position_embeddings = embedding.num_embeddings
            assert self.hidden_size == get_hidden_size(
                embedding
            ), "vocab embedding and positional embedding hidden_size does not match"
            self.positional_embedding = build_embedding(embedding, dtype=self.dtype)
        else:
            raise NotImplementedError(f"{embedding} not supported")

    def add_decoder_layers(self, input_decoder_layers):
        """Adds the transformer decoding layers to the model from the decoders as nn.ModuleList."""
        decoder_layers = []
        self.num_layers = len(input_decoder_layers)
        for layer in input_decoder_layers:
            trt_layer = build_decoder_layer(
                layer,
                self.num_layers,
                dtype=self.dtype,
                rank=self.rank,
                tensor_parallel=self.tensor_parallel,
            )
            if self.max_position_embeddings != 0:
                assert self.max_position_embeddings == trt_layer.max_position_embeddings
            else:
                self.max_position_embeddings == trt_layer.max_position_embeddings
            self.num_attention_heads = trt_layer.num_attention_heads
            self.hidden_act = trt_layer.hidden_act
            decoder_layers.append(trt_layer)
        self.layers = ModuleList(decoder_layers)

    def add_final_layernorm(self, layer_norm: nn.ModuleList):
        """Adds the final layernorm layer."""
        self.ln_f = build_layernorm(layer_norm, dtype=self.dtype)

    def load_nemo(self, weights_dir: Path, model_config: GPT2Config):
        """Loads the nemo model from the exported weights_dir into the TRT LLM network."""
        self.vocab_size = model_config.vocab_size
        self.hidden_size = model_config.n_embd
        self.max_position_embeddings = model_config.n_positions
        self.vocab_embedding = Embedding(self.vocab_size, self.hidden_size, dtype=self.dtype)
        self.vocab_embedding.weight.value = get_tensor_from_file(
            weights_dir, "wte", shape=[self.vocab_size, self.hidden_size], dtype=self.dtype
        )

        assert model_config.rotary_pct > 0, "GPT Next uses rotary embedding."

        self.num_layers = model_config.n_layer
        input_decoder_layers = [NemoLayer(weights_dir, i, model_config) for i in range(self.num_layers)]
        self.add_decoder_layers(input_decoder_layers)

        self.ln_f = LayerNorm(normalized_shape=self.hidden_size, dtype=self.dtype)
        self.ln_f.weight.value = get_tensor_from_file(weights_dir, "final_layernorm.weight", dtype=self.dtype)
        self.ln_f.bias.value = get_tensor_from_file(weights_dir, "final_layernorm.bias", dtype=self.dtype)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        past_key_value=None,
        sequence_length=None,
        past_key_value_length=None,
        masked_tokens=None,
        use_cache=False,
        attention_mask=None,
        cache_indirection=None,
    ):
        """Forward function for the full model."""
        x = self.vocab_embedding(input_ids.data)
        if hasattr(self, "positional_embedding"):
            assert position_ids
            x = x + self.positional_embedding(position_ids)

        hidden_states = x

        if past_key_value is None:
            past_key_value = tuple([None] * len(self.layers))

        if use_cache:
            presents = []

        if attention_mask is not None:
            attention_mask = expand_mask(attention_mask, shape(input_ids.data, -1))
        hidden_states = RaggedTensor.from_row_lengths(hidden_states, input_ids.row_lengths, input_ids.max_row_length)

        for layer, past in zip(self.layers, past_key_value):
            hidden_states = layer(
                hidden_states,
                past_key_value=past,
                sequence_length=sequence_length,
                past_key_value_length=past_key_value_length,
                masked_tokens=masked_tokens,
                use_cache=use_cache,
                attention_mask=attention_mask,
                cache_indirection=cache_indirection,
            )

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        hidden_states = self.ln_f(hidden_states.data)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class LMHeadModelBuilder(ModelBuilder):
    """The implementation of the model builder with an LMHead."""

    def __init__(self, dtype: trt.DataType = trt.float16, rank: int = 0, tensor_parallel: int = 1):
        super().__init__(dtype, rank, tensor_parallel)

    def finalize(self, lm_head: nn.Module = None):
        """Finalizes the LMHead model.

        If lm_head is not provided, we use the vocab embedding as the lm_head in tensorrt_llm."""
        vocab_size_padded = pad_vocab_size(self.vocab_size, self.tensor_parallel)
        self.lm_head = ColumnLinear(
            self.hidden_size,
            vocab_size_padded,
            bias=False,
            dtype=self.dtype,
            tp_group=get_tensor_parallel_group(self.tensor_parallel),
            tp_size=self.tensor_parallel,
            gather_output=True,
        )

        if lm_head:
            lm_head_weight = torch_to_np(lm_head.weight, self.dtype)
        else:
            # We use wte weights if not provided.
            lm_head_weight = self.vocab_embedding.weight._value

        if vocab_size_padded != self.vocab_size:
            pad_width = vocab_size_padded - self.vocab_size
            lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)), "constant", constant_values=0)

        self.lm_head.weight.value = np.ascontiguousarray(split(lm_head_weight, self.tensor_parallel, self.rank))

    def load_nemo(self, weights_dir: Path, model_config: GPT2Config):
        super().load_nemo(weights_dir, model_config)
        vocab_size_padded = pad_vocab_size(self.vocab_size, self.tensor_parallel)
        self.lm_head = ColumnLinear(
            self.hidden_size,
            vocab_size_padded,
            bias=False,
            dtype=self.dtype,
            tp_group=get_tensor_parallel_group(self.tensor_parallel),
            tp_size=self.tensor_parallel,
            gather_output=True,
        )

        lm_head_weight = get_tensor_from_file(
            weights_dir,
            "lm_head.weight",
            shape=[self.vocab_size, model_config.n_embd],
            dtype=self.dtype,
        )

        if vocab_size_padded != self.vocab_size:
            pad_width = vocab_size_padded - self.vocab_size
            lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)), "constant", constant_values=0)

        self.lm_head.weight.value = np.ascontiguousarray(split(lm_head_weight, self.tensor_parallel, self.rank))

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        past_key_value=None,
        sequence_length=None,
        past_key_value_length=None,
        masked_tokens=None,
        use_cache=False,
        last_token_ids=None,
        attention_mask=None,
        cache_indirection=None,
    ):
        """Forward function for the full LMHead model."""
        hidden_states = super().forward(
            input_ids,
            position_ids,
            past_key_value,
            sequence_length,
            past_key_value_length,
            masked_tokens,
            use_cache,
            attention_mask,
            cache_indirection,
        )

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = gather_last_token_logits(
            hidden_states, last_token_ids, default_net().plugin_config.remove_input_padding
        )

        # [batch_size, hidden_size] -> [batch_size, vocab_size]
        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output("logits", self.dtype)
        # out_inter.mark_output('inter', str_dtype_to_trt('float32'))

        if use_cache:
            for i, present in enumerate(presents):
                k, v = present
                k.mark_output(f"present_key_{i}", self.dtype)
                v.mark_output(f"present_value_{i}", self.dtype)
            return (lm_logits, presents)

        return lm_logits

    def prepare_inputs(self, max_batch_size, max_input_len, max_new_tokens, use_cache, max_beam_width: int = 1):
        """@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
        ranges of the dimensions of when using TRT dynamic shapes.

        @return: a list contains values which can be fed into the self.forward()
        """

        # Prepare inputs
        head_size = self.hidden_size // self.num_attention_heads
        num_heads = self.num_attention_heads // self.tensor_parallel
        num_heads_kv = 1 if self.multi_query_mode else num_heads
        max_len = max_input_len + max_new_tokens
        bb_range = [1, (max_batch_size * max_beam_width + 1) // 2, max_batch_size * max_beam_width]
        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        beam_width_range = [1, (max_beam_width + 1) // 2, max_beam_width]
        inlen_range = [1, 1, max_input_len]
        max_len_range = [0, (max_len + 1) // 2, max_len]
        mask_len_range = [1, (max_len + 1) // 2 + 1, max_len + 1]
        num_tokens_range = [
            1,
            max_batch_size * max_beam_width,
            max(max_input_len * max_batch_size, max_beam_width * max_batch_size),
        ]

        past_key_value = []
        sequence_length = None
        past_key_value_length = None
        masked_tokens = None
        attention_mask = None
        use_gpt_attention_plugin = default_net().plugin_config.gpt_attention_plugin
        remove_input_padding = default_net().plugin_config.remove_input_padding

        if remove_input_padding:
            input_ids = Tensor(
                name="input_ids",
                dtype=trt.int32,
                shape=[1, -1],
                dim_range=OrderedDict(
                    [
                        ("batch_size", [1]),
                        ("num_tokens", [num_tokens_range]),
                    ]
                ),
            )
            position_ids = Tensor(
                name="position_ids",
                dtype=trt.int32,
                shape=[1, -1],
                dim_range=OrderedDict(
                    [
                        ("batch_size", [1]),
                        ("num_tokens", [num_tokens_range]),
                    ]
                ),
            )
        else:
            input_ids = Tensor(
                name="input_ids",
                dtype=trt.int32,
                shape=[-1, -1],
                dim_range=OrderedDict(
                    [
                        ("batch_size", [bb_range]),
                        ("input_len", [inlen_range]),
                    ]
                ),
            )
            position_ids = Tensor(
                name="position_ids",
                dtype=trt.int32,
                shape=[-1, -1],
                dim_range=OrderedDict(
                    [
                        ("batch_size", [bb_range]),
                        ("input_len", [inlen_range]),
                    ]
                ),
            )

        for i in range(self.num_layers):
            kv_dim_range = OrderedDict(
                [
                    ("batch_size", [bb_range]),
                    ("num_heads", [num_heads_kv]),
                    ("past_key_len", [max_len_range]),
                    ("head_size", [head_size]),
                ]
            )
            k = Tensor(
                name=f"past_key_{i}",
                dtype=self.kv_dtype,
                shape=[-1, num_heads_kv, -1, head_size],
                dim_range=kv_dim_range,
            )
            v = Tensor(
                name=f"past_value_{i}",
                dtype=self.kv_dtype,
                shape=[-1, num_heads_kv, -1, head_size],
                dim_range=kv_dim_range,
            )
            past_key_value.append((k, v))

            # TODO(kaiyu): Remove this when TRT fix the named dimension
            if not remove_input_padding:
                assertion(shape(input_ids, 0) == shape(k, 0), "batch size")
                assertion(shape(input_ids, 0) == shape(v, 0), "batch size")
            assertion(shape(k, 2) == shape(v, 2), "kv cache len")

        if use_gpt_attention_plugin:
            sequence_length = Tensor(
                name="sequence_length",
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([("batch_size", [bb_range])]),
            )
            past_key_value_length = Tensor(
                name="past_key_value_length",
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([("past_key_value_length", [max_len_range])]),
            )
            masked_tokens = Tensor(
                name="masked_tokens",
                dtype=trt.int32,
                shape=[-1, -1],
                dim_range=OrderedDict(
                    [
                        ("batch_size", [bb_range]),
                        ("max_seq_len", [max_len_range]),
                    ]
                ),
            )
        else:
            attention_mask = Tensor(
                name="attention_mask",
                dtype=trt.int32,
                shape=[-1, -1],
                dim_range=OrderedDict(
                    [
                        ("batch_size", [bb_range]),
                        ("mask_len", [mask_len_range]),
                    ]
                ),
            )

        input_lengths = Tensor(
            name="input_lengths",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([("batch_size", [bb_range])]),
        )

        max_input_length = Tensor(
            name="max_input_length",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([("input_len", [inlen_range])]),
        )

        last_token_ids = Tensor(
            name="last_token_ids",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict(
                [
                    ("batch_size", [bb_range]),
                ]
            ),
        )
        input_ids_ragged = RaggedTensor.from_row_lengths(input_ids, input_lengths, max_input_length)

        cache_indirection = Tensor(
            name="cache_indirection",
            dtype=trt.int32,
            shape=[-1, -1, -1],
            dim_range=OrderedDict(
                [
                    ("batch_size", [bs_range]),
                    ("beam_width", [beam_width_range]),
                    ("max_seq_len", [max_len_range]),
                ]
            ),
        )

        return (
            input_ids_ragged,
            position_ids,
            past_key_value,
            sequence_length,
            past_key_value_length,
            masked_tokens,
            True,
            last_token_ids,
            attention_mask,
            cache_indirection,
        )

    def build(
        self,
        dtype: str = "float16",
        timing_cache: str = "model.cache",
        log_level: str = "info",
        max_batch_size: int = 1,
        max_input_len: int = 200,
        max_output_len: int = 200,
        max_beam_width: int = 1,
        parallel_build: bool = False,
        output_dir: str = "/tmp/ammo/",
    ):
        """Builds the model and generate the tensorrt_llm engine.

        Args:
            dtype: the deployed engine data type. Can be float16 or float32.
            timing_cache: the name of the tensorrt timing cache file inside the output_dir.
            log_level: the logging level.
            max_batch_size: the max batch size of the deployed model engine.
            max_input_len: the max length of the input tokens.
            max_output_len: the max length of the output tokens.
            output_dir: the output directory where we save the generated tensorrt_llm engine file.
        """
        assert self.rank < torch.cuda.device_count(), f"Rank {self.rank} out of bound"
        assert self.tensor_parallel <= torch.cuda.device_count(), f"Not enough GPUs, requesting {self.tensor_parallel}"

        build(
            self,
            self.rank,
            self.tensor_parallel,
            dtype,
            timing_cache,
            log_level,
            max_batch_size,
            max_input_len,
            max_output_len,
            max_beam_width,
            parallel_build,
            torch.cuda.device_count(),
            output_dir,
        )
