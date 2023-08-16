# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Referrence impl https://gitlab-master.nvidia.com/ftp/tekit/-/blob/main/examples/gpt/build.py"""

import argparse
import os
import time

import tensorrt_llm
import torch
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode

MODEL_NAME = "ammo"


def get_engine_name(model, dtype, tp_size, rank):
    return "{}_{}_tp{}_rank{}.engine".format(model, dtype, tp_size, rank)


def serialize_engine(engine, path):
    logger.info(f"Serializing engine to {path}...")
    tik = time.time()
    with open(path, "wb") as f:
        f.write(bytearray(engine))
    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    logger.info(f"Engine serialized. Total time: {t}")


def build_rank_engine(
    tensorrt_llm_gpt,
    builder: Builder,
    builder_config: tensorrt_llm.builder.BuilderConfig,
    engine_name,
    rank,
    args,
):
    """
    @brief: Build the engine on the given rank.
    @param rank: The rank to build the engine.
    @param args: The cmd line arguments.
    @return: The built engine.
    """
    str_dtype_to_trt(args.dtype)

    # TODO: Enable share_embedding_table
    # # Decide if we can share the embedding table between
    # # the lookup OP and the logits calculation OP
    # share_embedding_table = False
    # if args.use_lookup_plugin and args.model_dir is not None:
    #     share_embedding_table = check_embedding_share(args.model_dir)
    share_embedding_table = None

    if share_embedding_table and (not args.use_gemm_plugin):
        logger.warning(
            "Sharing embedding tables between OPs requires using GEMM plugin. Otherwise, you might"
            " fail to see the engine size reduction."
        )

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(dtype=args.use_gpt_attention_plugin)
    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
    if args.use_layernorm_plugin:
        network.plugin_config.set_layernorm_plugin(dtype=args.use_layernorm_plugin)
    assert not (args.enable_context_fmha and args.enable_context_fmha_fp32_acc)
    if args.enable_context_fmha:
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
    if args.enable_context_fmha_fp32_acc:
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled_with_fp32_acc)
    if args.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()
    if args.paged_kv_cache:
        network.plugin_config.enable_paged_kv_cache()
    if args.use_inflight_batching:
        network.plugin_config.set_inflight_batching_gpt_attention_plugin(
            dtype=args.use_inflight_batching
        )

    if args.world_size > 1:
        network.plugin_config.set_nccl_plugin(args.dtype)

    if args.use_lookup_plugin:
        # Use the plugin for the embedding parallelism and sharing
        network.plugin_config.set_lookup_plugin(dtype=args.dtype)
    assert not (
        args.use_lookup_plugin and args.max_prompt_embedding_table_size > 0
    ), "Lookup plugin isn't compatible with prompt tuning right now"

    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_gpt.named_parameters())

        # Forward
        inputs = tensorrt_llm_gpt.prepare_inputs(
            args.max_batch_size,
            args.max_input_len,
            args.max_output_len,
            True,
            args.max_beam_width,
            paged_kv_cache=args.paged_kv_cache,
            tokens_per_block=args.tokens_per_block,
            prompt_embedding_table_size=args.max_prompt_embedding_table_size,
        )
        tensorrt_llm_gpt(*inputs)

    engine = None

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    if rank == 0:
        config_path = os.path.join(args.output_dir, "config.json")
        builder.save_config(builder_config, config_path)
    return engine


def _build_impl(rank, tensorrt_llm_model, args):
    torch.cuda.set_device(rank % args.gpus_per_node)
    tensorrt_llm.logger.set_level(args.log_level)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    builder = Builder()
    cache = None
    apply_query_key_layer_scaling = False
    cur_rank = rank

    builder_config = builder.create_builder_config(
        name=MODEL_NAME,
        precision=args.dtype,
        timing_cache=args.timing_cache if cache is None else cache,
        tensor_parallel=args.world_size,  # TP only
        parallel_build=args.parallel_build,
        num_layers=tensorrt_llm_model._num_layers,
        num_heads=tensorrt_llm_model._num_heads,
        hidden_size=tensorrt_llm_model._hidden_size,
        vocab_size=tensorrt_llm_model._vocab_size,
        hidden_act=tensorrt_llm_model.hidden_act,
        max_position_embeddings=tensorrt_llm_model.max_position_embeddings,
        apply_query_key_layer_scaling=apply_query_key_layer_scaling,
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        int8="int8" in args.quantization,
        opt_level=args.builder_opt,
        multi_query_mode=args.multi_query_mode,
        paged_kv_cache=args.paged_kv_cache,
        tokens_per_block=args.tokens_per_block,
        use_prompt_tuning=args.max_prompt_embedding_table_size > 0,
        use_parallel_embedding=bool(args.use_lookup_plugin),
        fp8="fp8" in args.quantization,
    )

    engine_name = get_engine_name(MODEL_NAME, args.dtype, args.world_size, cur_rank)
    engine = build_rank_engine(
        tensorrt_llm_model, builder, builder_config, engine_name, cur_rank, args
    )
    assert engine is not None, f"Failed to build engine for rank {cur_rank}"

    if cur_rank == 0:
        # Use in-memory timing cache for multiple builder passes.
        if not args.parallel_build:
            cache = builder_config.trt_builder_config.get_timing_cache()

    serialize_engine(engine, os.path.join(args.output_dir, engine_name))

    if rank == 0:
        ok = builder.save_timing_cache(builder_config, os.path.join(args.output_dir, "model.cache"))
        assert ok, "Failed to save timing cache."


def build(
    tensorrt_llm_model,
    rank=0,
    world_size=1,
    dtype="float16",
    timing_cache="model.cache",
    log_level="info",
    max_batch_size=1,
    max_input_len=200,
    max_output_len=200,
    max_beam_width=1,
    max_prompt_embedding_table_size=0,
    parallel_build=False,
    gpus_per_node=1,
    output_dir="/tmp/ammo/",
    quantization=None,
):
    args = argparse.Namespace()
    args.world_size = world_size
    args.dtype = dtype
    args.timing_cache = timing_cache
    args.log_level = log_level
    args.max_batch_size = max_batch_size
    args.max_input_len = max_input_len
    args.max_output_len = max_output_len
    args.max_beam_width = max_beam_width
    args.use_gpt_attention_plugin = dtype
    args.use_gemm_plugin = dtype
    args.use_layernorm_plugin = False
    args.parallel_build = parallel_build
    args.enable_context_fmha = True
    args.enable_context_fmha_fp32_acc = False
    args.gpus_per_node = gpus_per_node
    args.builder_opt = None
    args.output_dir = output_dir
    args.multi_query_mode = False
    args.remove_input_padding = False
    args.use_smooth_quant = False
    args.use_weight_only = False
    args.weight_only_precision = "int8"
    args.per_channel = False
    args.per_token = False
    args.int8_kv_cache = False
    args.random_seed = None
    args.paged_kv_cache = False
    args.max_prompt_embedding_table_size = max_prompt_embedding_table_size
    args.use_inflight_batching = False
    args.use_lookup_plugin = False
    args.tokens_per_block = 64
    args.quantization = quantization

    assert not (
        args.use_smooth_quant and args.use_weight_only
    ), "You cannot enable both SmoothQuant and INT8 weight-only together."

    if args.use_smooth_quant:
        args.quant_mode = QuantMode.use_smooth_quant(args.per_token, args.per_channel)
    elif args.use_weight_only:
        args.quant_mode = QuantMode.use_weight_only(args.weight_only_precision == "int4")
    else:
        args.quant_mode = QuantMode(0)

    if args.int8_kv_cache:
        args.quant_mode = args.quant_mode.set_int8_kv_cache()

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)

    logger.set_level(args.log_level)
    tik = time.time()
    _build_impl(rank, tensorrt_llm_model, args)

    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    logger.info(f"Total time of building all {args.world_size} engines: {t}")
