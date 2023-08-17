# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Referrence impl https://gitlab-master.nvidia.com/ftp/tekit/-/blob/main/tensorrt_llm/models/gpt/model.py"""

import math
import os
from collections import OrderedDict

import numpy as np
import tensorrt as trt
import torch
from tensorrt_llm._common import default_net
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.functional import (
    RaggedTensor,
    Tensor,
    assertion,
    expand_mask,
    gather_last_token_logits,
    shape,
)
from tensorrt_llm.layers import ColumnLinear, InflightBatchingParam
from tensorrt_llm.module import Module, ModuleList

from .decoder import build_decoder_layer
from .model_config import QUANTIZATION_FP8, QUANTIZATION_INT8_SQ, ModelConfig
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
        super().__init__()
        self.quantization = model_config.quantization
        self.rank = model_config.rank
        self.max_position_embeddings = model_config.max_position_embeddings
        self.hidden_act = model_config.hidden_act

        self._dtype = str_dtype_to_trt(model_config.dtype)
        self._kv_dtype = self._dtype
        self._tensor_parallel = model_config.tensor_parallel
        self._multi_query_mode = False
        self._vocab_size = model_config.vocab_size
        self._hidden_size = model_config.hidden_size
        self._num_layers = len(model_config.layers)
        self._num_heads = model_config.num_attention_heads
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
                    self._num_layers,
                    dtype=self._dtype,
                    quantization=model_config.quantization,
                    rank=self.rank,
                    tensor_parallel=self._tensor_parallel,
                )
                for layer in model_config.layers
            ]
        )

        self.ln_f = build_layernorm_from_config(model_config.final_layernorm, self._dtype)

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
        kv_cache_block_pointers=None,
        prompt_embedding_table=None,
        prompt_tasks=None,
        prompt_vocab_size=None,
        inflight_batching_args=None,
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
        for idx, (layer, past, pointers) in enumerate(
            zip(self.layers, past_key_value, kv_cache_block_pointers)
        ):
            hidden_states = layer(
                hidden_states,
                past_key_value=past,
                sequence_length=sequence_length,
                past_key_value_length=past_key_value_length,
                masked_tokens=masked_tokens,
                use_cache=use_cache,
                attention_mask=attention_mask,
                cache_indirection=cache_indirection,
                kv_cache_block_pointers=pointers,
                inflight_batching_args=inflight_batching_args,
                past_key_value_pointers=(
                    None
                    if inflight_batching_args is None
                    else inflight_batching_args.past_key_value_pointers[idx]
                ),
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

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        # TODO: Add support for share_embedding_table
        share_embedding_table = False
        share_weight = None
        if share_embedding_table is True:
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
        if model_config.quantization == QUANTIZATION_FP8:
            self.lm_head = quantize_linear(self.lm_head, QUANTIZATION_FP8, model_config.lm_head)

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
        kv_cache_block_pointers=None,
        prompt_embedding_table=None,
        prompt_tasks=None,
        prompt_vocab_size=None,
        inflight_batching_args=None,
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
            kv_cache_block_pointers,
            prompt_embedding_table,
            prompt_tasks,
            prompt_vocab_size,
            inflight_batching_args,
        )

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = gather_last_token_logits(
            hidden_states, last_token_ids, default_net().plugin_config.remove_input_padding
        )

        # [batch_size, hidden_size] -> [batch_size, vocab_size]
        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output("logits", self._dtype)
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
        """@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
        ranges of the dimensions of when using TRT dynamic shapes.

        @return: a list contains values which can be fed into the self.forward()
        """

        # Prepare inputs
        head_size = self._hidden_size // self._num_heads
        num_heads = self._num_heads // self._tensor_parallel
        num_heads_kv = 1 if self._multi_query_mode else num_heads
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
        if paged_kv_cache:
            blocks_range = [
                math.ceil((bb_range[0] * max_len_range[0]) / tokens_per_block),
                math.ceil((bb_range[1] * max_len_range[1]) / tokens_per_block),
                math.ceil((bb_range[2] * max_len_range[2]) / tokens_per_block),
            ]
            # NOTE(nkorobov): we multiply max_blocks_per_seq by 2 because plugin expects pointers as int64,
            # but TRT does not support int64. Thus, we emulate int64 with doubled int32.
            max_blocks_per_seq_range = [
                2 * math.ceil(max_len_range[0] / tokens_per_block),
                2 * math.ceil(max_len_range[1] / tokens_per_block),
                2 * math.ceil(max_len_range[2] / tokens_per_block),
            ]
        p_embedding_range = [1, prompt_embedding_table_size // 2, prompt_embedding_table_size]

        past_key_value = []
        past_key_value_pointers = []
        sequence_length = None
        past_key_value_length = None
        masked_tokens = None
        attention_mask = None
        inflight_batching_args = None
        cache_indirection = None
        use_gpt_attention_plugin = default_net().plugin_config.gpt_attention_plugin
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_inflight_batching_gpt_attention_plugin = (
            default_net().plugin_config.inflight_batching_gpt_attention_plugin
        )

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
            tasks = Tensor(
                name="tasks",
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict(
                    [
                        ("batch_size", [bb_range]),
                    ]
                ),
            )
            prompt_vocab_size = Tensor(
                name="prompt_vocab_size",
                dtype=trt.int32,
                shape=[1],
                dim_range=OrderedDict([("size", [1])]),
            )

        for i in range(self._num_layers):
            if not paged_kv_cache:
                kv_dim_range = OrderedDict(
                    [
                        ("batch_size", [bb_range]),
                        ("kv", [2]),
                        ("num_heads", [num_heads_kv]),
                        ("past_key_len", [max_len_range]),
                        ("head_size", [head_size]),
                    ]
                )
                kv = Tensor(
                    name=f"past_key_value_{i}",
                    dtype=self._kv_dtype,
                    shape=[-1, 2, num_heads_kv, -1, head_size],
                    dim_range=kv_dim_range,
                )
                past_key_value.append(kv)

                # TODO(kaiyu): Remove this when TRT fix the named dimension
                if not remove_input_padding:
                    assertion(shape(input_ids, 0) == shape(kv, 0), "batch size")
            else:
                kv_dim_range = OrderedDict(
                    [
                        ("blocks", [blocks_range]),
                        ("kv", [2]),
                        ("num_heads", [num_heads_kv]),
                        ("tokens_per_block", [tokens_per_block]),
                        ("head_size", [head_size]),
                    ]
                )
                # (2, blocks, kv_num_heads, tokens_per_block, head_size)
                kv = Tensor(
                    name=f"past_key_value_{i}",
                    dtype=self._kv_dtype,
                    shape=[-1, 2, num_heads_kv, tokens_per_block, head_size],
                    dim_range=kv_dim_range,
                )
                past_key_value.append(kv)

            if use_inflight_batching_gpt_attention_plugin:
                kv = Tensor(
                    name=f"past_key_value_pointers_{i}",
                    dtype=trt.int32,
                    # 2 INT32s for representing a single INT64 pointer
                    shape=[-1, 2],
                    dim_range=OrderedDict(batch_size=[bs_range], pointer_width=[2]),
                )

                past_key_value_pointers.append(kv)

        if use_gpt_attention_plugin or use_inflight_batching_gpt_attention_plugin:
            past_key_value_length = Tensor(
                name="past_key_value_length",
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([("past_key_value_length", [max_len_range])]),
            )

        if use_gpt_attention_plugin:
            sequence_length = Tensor(
                name="sequence_length",
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([("batch_size", [bb_range])]),
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
        elif not use_inflight_batching_gpt_attention_plugin:
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
            dim_range=OrderedDict([("max_input_len", [inlen_range])]),
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

        if not use_inflight_batching_gpt_attention_plugin:
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

        if use_inflight_batching_gpt_attention_plugin:
            inflight_batching_args = InflightBatchingParam(
                # [nbReq]
                host_input_lengths=Tensor(
                    name="host_input_lengths",
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict(batch_size=[bs_range]),
                ),
                # [nbReq]
                host_beam_widths=Tensor(
                    name="beam_widths",
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict(batch_size=[bs_range]),
                ),
                # [nbReq, 2]
                cache_indir_pointers=Tensor(
                    name="cache_indir_pointers",
                    dtype=trt.int32,
                    # 2 INT32s for representing a single INT64 pointer
                    shape=[-1, 2],
                    dim_range=OrderedDict(batch_size=[bs_range], pointer_width=[2]),
                ),
                # [nbReq]
                host_req_cache_max_seq_lengths=Tensor(
                    name="req_cache_max_seq_lengths",
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict(batch_size=[bs_range]),
                ),
                max_input_length=max_input_len,
                max_beam_width=max_beam_width,
                use_int8_kv_cache=self.quant_mode.has_int8_kv_cache(),
                past_key_value_pointers=past_key_value_pointers,
            )

        kv_cache_block_pointers_list = []
        for i in range(self._num_layers):
            if paged_kv_cache:
                kv_cache_block_pointers = Tensor(
                    name=f"kv_cache_block_pointers_{i}",
                    dtype=trt.int32,
                    shape=[-1, -1, 2, -1],
                    dim_range=OrderedDict(
                        [
                            ("batch_size", [bs_range]),
                            ("beam_width", [beam_width_range]),
                            ("kv", [2]),
                            ("max_blocks_per_seq", [max_blocks_per_seq_range]),
                        ]
                    ),
                )
            else:
                kv_cache_block_pointers = None
            kv_cache_block_pointers_list.append(kv_cache_block_pointers)

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
            kv_cache_block_pointers_list,
            prompt_embedding_table,
            tasks,
            prompt_vocab_size,
            inflight_batching_args,
        )

    def build(
        self,
        timing_cache: str = "model.cache",
        log_level: str = "info",
        max_batch_size: int = 1,
        max_input_len: int = 200,
        max_output_len: int = 200,
        max_beam_width: int = 1,
        parallel_build: bool = False,
        max_prompt_embedding_table_size: int = 0,
        output_dir: str = "/tmp/ammo/",
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

        assert self.rank < torch.cuda.device_count(), f"Rank {self.rank} out of bound"
        assert (
            self._tensor_parallel <= torch.cuda.device_count()
        ), f"Not enough GPUs, requesting {self._tensor_parallel}"

        if self.quantization == QUANTIZATION_FP8 or self.quantization == QUANTIZATION_INT8_SQ:
            # A hot fix to TRT for performance improvement.
            os.environ["__LUNOWUD"] = "-peep:transpose_elim=off"

        self._use_prompt_tuning = max_prompt_embedding_table_size > 0
        build(
            tensorrt_llm_model=self,
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
            output_dir=output_dir,
            quantization=self.quantization,
        )

    def print(self):
        """Debugging print of the tensorrt_llm network."""
        np.set_printoptions(threshold=36)
        print_tensorrt_llm(f"rank.{self.rank}", self)
