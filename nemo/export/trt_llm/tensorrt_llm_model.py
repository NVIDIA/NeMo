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
    RaggedTensor,
    Tensor,
    expand_mask,
    gather_last_token_logits,
    shape,
)
from tensorrt_llm.layers import ColumnLinear, InflightBatchingParam
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
        self._use_prompt_tuning = False

        # TODO: support use_prompt_tuning and use_parallel_embedding.
        self.vocab_embedding = build_embedding_from_config(
            model_config.vocab_embedding, self._dtype
        )
        self.positional_embedding = build_embedding_from_config(
            model_config.positional_embedding, self._dtype
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
        # TODO: support use_prompt_tuning
        x = self.vocab_embedding(input_ids.data)
        if hasattr(self, "positional_embedding") and self.positional_embedding:
            assert position_ids
            x = x + self.positional_embedding(position_ids)

        hidden_states = x

        if past_key_value is None:
            past_key_value = tuple([None] * len(self.layers))

        if use_cache:
            presents = []

        if attention_mask is not None:
            attention_mask = expand_mask(attention_mask, shape(input_ids.data, -1))
        hidden_states = RaggedTensor.from_row_lengths(
            hidden_states, input_ids.row_lengths, input_ids.max_row_length
        )

        def _forward_has_argument(layer, argument_name):
            return argument_name in inspect.signature(layer.forward).parameters

        for idx, (layer, past, pointers) in enumerate(
            zip(self.layers, past_key_value, kv_cache_block_pointers)
        ):
            # In TRT LLM, not all model decoders are with the same forward arg signature.
            # So we check arg compatibility and optionally add them if supported.
            # In case the decoder forward signature changes, this if branch list below will need to be updated.
            additional_inputs = {}
            if _forward_has_argument(layer, "inflight_batching_args"):
                additional_inputs["inflight_batching_args"] = inflight_batching_args
            if _forward_has_argument(layer, "past_key_value_pointers"):
                additional_inputs["past_key_value_pointers"] = (
                    (
                        None
                        if inflight_batching_args is None
                        else inflight_batching_args.past_key_value_pointers[idx]
                    ),
                )
            if _forward_has_argument(layer, "pointers_to_kv_cache_block_pointers"):
                additional_inputs["pointers_to_kv_cache_block_pointers"] = (
                    (
                        None
                        if (
                            inflight_batching_args is None
                            or inflight_batching_args.pointers_to_kv_cache_block_pointers is None
                        )
                        else inflight_batching_args.pointers_to_kv_cache_block_pointers[idx]
                    ),
                )

            hidden_states = layer(
                hidden_states,
                past_key_value=past,
                sequence_length=sequence_length,
                host_past_key_value_lengths=host_past_key_value_lengths,
                use_cache=use_cache,
                attention_mask=attention_mask,
                cache_indirection=cache_indirection,
                kv_cache_block_pointers=pointers,
                context_lengths=context_lengths,
                host_context_lengths=host_context_lengths,
                host_request_types=host_request_types,
                max_context_length=max_context_length,
                **additional_inputs,
            )

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        hidden_states = self.ln_f(hidden_states.data)

        if use_cache:
            return (hidden_states, tuple(presents))
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
        use_ib_gpt_attention_plugin = (
            default_net().plugin_config.inflight_batching_gpt_attention_plugin
        )

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
            use_ib_gpt_attention_plugin=use_ib_gpt_attention_plugin,
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
            prompt_embedding_table = Tensor(
                name="prompt_embedding_table",
                dtype=self._dtype,
                shape=[-1, self._hidden_size],
                dim_range=OrderedDict(
                    [
                        ("prompt_embedding_table_size", [p_embedding_range]),
                        ("hidden_size", [self._hidden_size]),
                    ]
                ),
            )
            if remove_input_padding:
                tasks = Tensor(
                    name="tasks",
                    dtype=trt.int32,
                    shape=[1, -1],
                    dim_range=OrderedDict(
                        [
                            ("batch_size_fake", [1]),
                            ("input_len_task", [num_tokens_range]),
                        ]
                    ),
                )
            else:
                tasks = Tensor(
                    name="tasks",
                    dtype=trt.int32,
                    shape=[-1, -1],
                    dim_range=OrderedDict(
                        [
                            ("batch_size_beam_width", [bb_range]),
                            ("input_len_task", [inlen_range]),
                        ]
                    ),
                )
            prompt_vocab_size = Tensor(
                name="prompt_vocab_size",
                dtype=trt.int32,
                shape=[1],
                dim_range=OrderedDict([("size", [1])]),
            )

        inflight_batching_args = None
        if use_ib_gpt_attention_plugin:
            past_key_value_pointers = []
            pointers_to_kv_cache_block_pointers = []
            for i in range(self._num_layers):
                kv = Tensor(
                    name=f"past_key_value_pointers_{i}",
                    dtype=trt.int32,
                    # 2 INT32s for representing a single INT64 pointer
                    shape=[-1, 2],
                    dim_range=OrderedDict(batch_size_kv=[bs_range], pointer_width=[2]),
                )
                past_key_value_pointers.append(kv)

                if paged_kv_cache:
                    # [nbReq, 2]
                    pkv = Tensor(
                        name=f"pointers_to_kv_cache_block_pointers_{i}",
                        dtype=trt.int32,
                        # 2 INT32s for representing a single INT64 pointer
                        shape=[-1, 2],
                        dim_range=OrderedDict(batch_size_cp=[bs_range], pointer_width=[2]),
                    )
                    pointers_to_kv_cache_block_pointers.append(pkv)

            inflight_batching_args = InflightBatchingParam(
                # [nbReq]
                host_context_lengths=Tensor(
                    name="host_context_lengths",
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict(batch_size_hscl=[bs_range]),
                ),
                # [nbSeq]
                context_lengths=Tensor(
                    name="context_lengths",
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict(batch_size_context_lengths=[bs_range]),
                ),
                # [nbReq]
                host_beam_widths=Tensor(
                    name="beam_widths",
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict(batch_size_bw=[bs_range]),
                ),
                # [nbReq, 2]
                cache_indir_pointers=Tensor(
                    name="cache_indir_pointers",
                    dtype=trt.int32,
                    # 2 INT32s for representing a single INT64 pointer
                    shape=[-1, 2],
                    dim_range=OrderedDict(batch_size_cp=[bs_range], pointer_width=[2]),
                ),
                # [nbReq]
                host_req_cache_max_seq_lengths=Tensor(
                    name="req_cache_max_seq_lengths",
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict(batch_size_rcmsl=[bs_range]),
                ),
                max_input_length=max_input_len,
                max_beam_width=max_beam_width,
                use_int8_kv_cache=self.quant_mode.has_int8_kv_cache(),
                past_key_value_pointers=past_key_value_pointers,
                pointers_to_kv_cache_block_pointers=(
                    None if not paged_kv_cache else pointers_to_kv_cache_block_pointers
                ),
            )

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
            self,
            output_dir,
            self.rank,
            self._tensor_parallel,
            trt_dtype_to_str(self._dtype),
            timing_cache,
            log_level,
            max_batch_size,
            max_input_len,
            max_output_len,
            max_beam_width,
            parallel_build,
            torch.cuda.device_count(),
            quantization=self.quantization,
        )

    def print(self):
        """Debugging print of the tensorrt_llm network."""
        np.set_printoptions(threshold=36)
        print_tensorrt_llm(f"rank.{self.rank}", self)
