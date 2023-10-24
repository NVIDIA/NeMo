# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""This module defines a tensorrt_llm based model for all LLMs we support inside AMMO.

Referrence impl in tensorrt_llm: tensorrt_llm/models/gpt/model.py.
"""
import inspect
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from tensorrt_llm import default_net, str_dtype_to_trt
from tensorrt_llm.functional import (
    Tensor,
    expand_mask,
    gather_last_token_logits,
    shape,
)
from tensorrt_llm.layers import ColumnLinear, KeyValueCacheParams, AttentionParams
from tensorrt_llm.models.generation_mixin import GenerationMixin
from tensorrt_llm.module import Module, ModuleList

from .decoder import build_decoder_layer
from .model_config import ModelConfig
from .quantization_utils import quantize_linear
from .tensor_utils import (
    get_tensor_parallel_group,
    trt_dtype_to_str,
)
from .tensorrt_llm_build import build
from .tensorrt_llm_utils import (
    build_embedding_from_config,
    build_layernorm_from_config,
    print_tensorrt_llm,
)


class ModelBuilder(Module):
    """A generic tensorrt_llm transformer model builder.

    We try to make this module builder as flexibile as possible to cover all transformer conversion usecases.
    """

    def __init__(self, model_config: ModelConfig):
        """Initializes the ModelBuilder from a model_config."""
        super().__init__()
        self.quantization = model_config.quantization
        self.rank = model_config.rank
        self.max_position_embeddings = model_config.max_position_embeddings
        self.hidden_act = model_config.hidden_act

        self._dtype = str_dtype_to_trt(model_config.dtype)
        self._kv_dtype = self._dtype
        self._tensor_parallel = model_config.tensor_parallel
        self._vocab_size = model_config.vocab_size
        self._hidden_size = model_config.hidden_size
        self._num_layers = len(model_config.layers)
        self._num_heads = model_config.num_attention_heads
        self._num_kv_heads = model_config.num_kv_heads
        self._use_prompt_tuning = model_config.use_prompt_tuning

        # TODO: support use_parallel_embedding.
        self.vocab_embedding = build_embedding_from_config(
            model_config.vocab_embedding, self._dtype, use_prompt_tuning=self._use_prompt_tuning
        )
        self.positional_embedding = build_embedding_from_config(
            model_config.positional_embedding, self._dtype, use_prompt_tuning=False
        )
        self.layers = ModuleList(
            [
                build_decoder_layer(
                    layer,
                    layer_id,
                    self._num_layers,
                    dtype=self._dtype,
                    quantization=model_config.quantization,
                    rank=self.rank,
                    tensor_parallel=self._tensor_parallel,
                )
                for layer_id, layer in enumerate(model_config.layers)
            ]
        )

        self.ln_f = build_layernorm_from_config(model_config.final_layernorm, self._dtype)

    def forward(
        self,
        input_ids,
        position_ids,
        past_key_value=None,
        sequence_length=None,
        host_past_key_value_lengths=None,
        use_cache=False,
        attention_mask=None,
        cache_indirection=None,
        kv_cache_block_pointers=None,
        prompt_embedding_table=None,
        prompt_tasks=None,
        prompt_vocab_size=None,
        inflight_batching_args=None,
        context_lengths=None,
        host_context_lengths=None,
        host_request_types=None,
        max_context_length=None,
    ):
        """Forward function for the full model."""
        ptuning_args = []
        if self._use_prompt_tuning:
            ptuning_args = [prompt_embedding_table, prompt_tasks, prompt_vocab_size]
        x = self.vocab_embedding(input_ids, *ptuning_args)
        if hasattr(self, "positional_embedding") and self.positional_embedding:
            assert position_ids
            x = x + self.positional_embedding(position_ids)

        hidden_states = x

        if past_key_value is None:
            past_key_value = tuple([None] * len(self.layers))

        if use_cache:
            presents = []

        if attention_mask is not None:
            attention_mask = expand_mask(attention_mask, shape(input_ids, -1))

        def _forward_has_argument(layer, argument_name):
            return argument_name in inspect.signature(layer.forward).parameters

        for idx, (layer, past, pointers) in enumerate(
            zip(self.layers, past_key_value, kv_cache_block_pointers)
        ):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=host_past_key_value_lengths,
                    kv_cache_block_pointers=pointers,
                    cache_indirection=cache_indirection
                ),
                attention_params=AttentionParams(
                    sequence_length=sequence_length,
                    context_lengths=context_lengths,
                    host_context_lengths=host_context_lengths,
                    max_context_length=max_context_length,
                    host_request_types=host_request_types,
                ),
            )

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return hidden_states, tuple(presents)
        return hidden_states


class LMHeadModelBuilder(ModelBuilder, GenerationMixin):
    """The implementation of the model builder with an LMHead."""

    def __init__(self, model_config: ModelConfig):
        """Initializes the LMHeadModelBuilder from a model_config."""
        super().__init__(model_config)

        # TODO: Add support for share_embedding_table
        share_embedding_table = False
        share_weight = None
        if share_embedding_table:
            share_weight = self.embedding.vocab_embedding.weight
        self.lm_head = ColumnLinear(
            self._hidden_size,
            model_config.vocab_size_padded,
            bias=False,
            dtype=self._dtype,
            tp_group=get_tensor_parallel_group(self._tensor_parallel),
            tp_size=self._tensor_parallel,
            gather_output=True,
            share_weight=share_weight,
        )
        self.lm_head.weight.value = model_config.lm_head.weight
        if model_config.quantization:
            self.lm_head = quantize_linear(
                self.lm_head, model_config.quantization, model_config.lm_head
            )

    def forward(
        self,
        input_ids,
        position_ids,
        past_key_value=None,
        sequence_length=None,
        host_past_key_value_lengths=None,
        use_cache=False,
        last_token_ids=None,
        attention_mask=None,
        cache_indirection=None,
        kv_cache_block_pointers=None,
        prompt_embedding_table=None,
        prompt_tasks=None,
        prompt_vocab_size=None,
        inflight_batching_args=None,
        context_lengths=None,
        host_context_lengths=None,
        host_request_types=None,
        max_context_length=None,
    ):
        """Forward function for the full LMHead model."""
        assert last_token_ids is not None, "Expecting last token ids to be not None"
        hidden_states = super().forward(
            input_ids,
            position_ids,
            past_key_value,
            sequence_length,
            host_past_key_value_lengths,
            use_cache,
            attention_mask,
            cache_indirection,
            kv_cache_block_pointers,
            prompt_embedding_table,
            prompt_tasks,
            prompt_vocab_size,
            inflight_batching_args,
            context_lengths,
            host_context_lengths,
            host_request_types,
            max_context_length,
        )

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = gather_last_token_logits(
            hidden_states, last_token_ids, default_net().plugin_config.remove_input_padding
        )

        # [batch_size, hidden_size] -> [batch_size, vocab_size]
        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output("logits", str_dtype_to_trt("float16"))
        # out_inter.mark_output('inter', str_dtype_to_trt('float32'))

        if use_cache:
            for i, present in enumerate(presents):
                present.mark_output(f"present_key_value_{i}", self._kv_dtype)
            return (lm_logits, presents)

        return lm_logits

    def prepare_inputs(
        self,
        max_batch_size,
        max_input_len,
        max_new_tokens,
        use_cache,
        max_beam_width: int = 1,
        paged_kv_cache: bool = False,
        tokens_per_block: int = 64,
        prompt_embedding_table_size: int = 128,
    ):
        """@brief: Prepare inputs Tensors for the model.

        The given sizes are used to determine the
        ranges of the dimensions of when using TRT dynamic shapes.

        @return: a list contains values which can be fed into the self.forward()
        """
        # Prepare inputs
        head_size = self._hidden_size // self._num_heads
        num_heads_kv = (self._num_kv_heads + self._tensor_parallel - 1) // self._tensor_parallel
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net().plugin_config.gpt_attention_plugin

        model_inputs = self.prepare_basic_inputs(
            max_batch_size,
            max_beam_width,
            max_input_len,
            max_new_tokens,
            num_heads_kv,
            head_size,
            self._num_layers,
            self._kv_dtype,
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
        )

        bb_range = [1, (max_batch_size * max_beam_width + 1) // 2, max_batch_size * max_beam_width]
        p_embedding_range = [1, prompt_embedding_table_size // 2, prompt_embedding_table_size]
        num_tokens_range = [
            1,
            max_batch_size * max_beam_width,
            max(max_input_len * max_batch_size, max_beam_width * max_batch_size),
        ]
        inlen_range = [1, 1, max_input_len]
        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]

        prompt_embedding_table = None
        tasks = None
        prompt_vocab_size = None
        if self._use_prompt_tuning:
            assert prompt_embedding_table_size is not None, "prompt_embedding_table_size cannot be None when self._use_prompt_tuning is True"
            _p_embedding_range = [
                1, prompt_embedding_table_size // 2, prompt_embedding_table_size
            ]
            if enable_two_optimization_profiles:
                p_embedding_range = [_p_embedding_range, _p_embedding_range]
            else:
                p_embedding_range = [_p_embedding_range]

            prompt_embedding_table = Tensor(
                name='prompt_embedding_table',
                dtype=dtype,
                shape=[-1, self._hidden_size],
                dim_range=OrderedDict([
                    ('prompt_embedding_table_size', p_embedding_range),
                    ('hidden_size', [self._hidden_size, self._hidden_size]
                     if enable_two_optimization_profiles else [self._hidden_size]),
                ]))
            if remove_input_padding:
                tasks = Tensor(
                    name='tasks',
                    dtype=trt.int32,
                    shape=[1, -1],
                    dim_range=OrderedDict([
                        ('batch_size_fake',
                         [1, 1] if enable_two_optimization_profiles else [1]),
                        ('input_len_task', num_tokens_range),
                    ]))
            else:
                tasks = Tensor(
                    name='tasks',
                    dtype=trt.int32,
                    shape=[-1, 1],
                    dim_range=OrderedDict([
                        ('batch_size_beam_width', bb_range),
                        ('broadcast_dim',
                         [1, 1] if enable_two_optimization_profiles else [1]),
                    ]))
            prompt_vocab_size = Tensor(
                name='prompt_vocab_size',
                dtype=trt.int32,
                shape=[1],
                dim_range=OrderedDict([
                    ('size',
                     [1, 1] if enable_two_optimization_profiles else [1])
                ]))


        # todo: we should remove this, but hesitant since no explicit argument names below.
        inflight_batching_args = None

        return (
            model_inputs["input_ids"],
            model_inputs["position_ids"],
            model_inputs["past_key_value"],
            model_inputs["sequence_length"],
            model_inputs["host_past_key_value_lengths"],
            True,
            model_inputs["last_token_ids"],
            model_inputs["attention_mask"],
            model_inputs["cache_indirection"],
            model_inputs["kv_cache_block_pointers_list"],
            prompt_embedding_table,
            tasks,
            prompt_vocab_size,
            inflight_batching_args,
            model_inputs["context_lengths"],
            model_inputs["host_context_lengths"],
            model_inputs["host_request_types"],
            max_input_len,
        )


    def build(
        self,
        output_dir: Path,
        timing_cache: str = "",
        log_level: str = "info",
        max_batch_size: int = 1,
        max_input_len: int = 200,
        max_output_len: int = 200,
        max_beam_width: int = 1,
        parallel_build: bool = False,
        max_prompt_embedding_table_size: int = 0,
    ):
        """Builds the model and generate the tensorrt_llm engine.

        Args:
            timing_cache: the name of the tensorrt timing cache file inside the output_dir.
            log_level: the logging level.
            max_batch_size: the max batch size of the deployed model engine.
            max_input_len: the max length of the input tokens.
            max_output_len: the max length of the output tokens.
            max_beam_width: the max beam search width.
            output_dir: the output directory where we save the generated tensorrt_llm engine file.
        """
        # Uncomment the following to print the network for debugging purpose.
        # self.print()

        if self.rank < torch.cuda.device_count():
            print(f"warning: Rank {self.rank} larger than GPUs available")
        if self._tensor_parallel < torch.cuda.device_count():
            print(f"warning: Not enough GPUs locally, requesting {self._tensor_parallel}")

        build(
            tensorrt_llm_model=self,
            output_dir=output_dir,
            rank=self.rank,
            world_size=self._tensor_parallel,
            dtype=trt_dtype_to_str(self._dtype),
            timing_cache=timing_cache,
            log_level=log_level,
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_beam_width=max_beam_width,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
            parallel_build=parallel_build,
            gpus_per_node=torch.cuda.device_count(),
            quantization=self.quantization,
        )

    def print(self):
        """Debugging print of the tensorrt_llm network."""
        np.set_printoptions(threshold=36)
        print_tensorrt_llm(f"rank.{self.rank}", self)
