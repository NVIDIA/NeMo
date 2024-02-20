# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The APIs to convert a nemo model checkpoint to tensorrt_llm."""

import argparse
import ast
import configparser
import copy
import datetime
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
import typing
from typing import Dict, List, Tuple

import numpy as np
import tensorrt_llm
from tensorrt_llm import str_dtype_to_trt
from transformers import GPT2Config, LlamaConfig, PretrainedConfig, PreTrainedTokenizer, AutoTokenizer

from .model_config import (
    LAYERNORM_DEFAULT,
    LAYERNORM_RMS,
    LINEAR_COLUMN,
    DecoderLayerConfig,
    EmbeddingConfig,
    LayernormConfig,
    LinearConfig,
    ModelConfig,
)
from .nemo.nemo import UnpackedNemoCheckpointDir, unpack_nemo_ckpt
from .nemo.nemo_ckpt_convert import build_tokenizer, convert_checkpoint, convert_dist_checkpoint
from .tensor_utils import get_tensor_from_dict, split

LOGGER = logging.getLogger("NeMo")


def _nemo_decode(
    in_file: str,
    out_dir: str,
    tensor_parallelism: int = 1,
    processes: int = 1,
    storage_type: str = "bfloat16",
    load_checkpoints_on_gpu: bool = False,
    decoder_type: str = "gptnext",
    save_nemo_model_config: bool = False,
) -> Tuple[Dict[str, np.ndarray], PretrainedConfig, PreTrainedTokenizer]:
    """Decodes the NEMO file and returns the weights dict, llm config and tokenizer."""
    args = argparse.Namespace()
    args.in_file = in_file
    args.out_dir = out_dir
    args.tensor_parallelism = tensor_parallelism
    args.processes = processes
    args.storage_type = storage_type
    args.load_checkpoints_on_gpu = load_checkpoints_on_gpu
    args.verbose = False
    args.decoder_type = decoder_type

    input_path = Path(args.in_file)
    if not input_path.exists():
        LOGGER.error("%s does not exists", input_path)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # unpack if needed
        if input_path.is_dir():
            nemo_dir = input_path
        else:
            start_time = datetime.datetime.now()
            checkpoint_dir_path = temp_dir / "unpacked"
            nemo_dir = unpack_nemo_ckpt(args.in_file, checkpoint_dir_path)
            LOGGER.info(
                "Spent %s (h:m:s) to unpack NeMo archive", datetime.datetime.now() - start_time
            )

        unpacked_checkpoint_dir = UnpackedNemoCheckpointDir(
            nemo_dir, load_checkpoints_to_cpu=not args.load_checkpoints_on_gpu
        )

        start_time = datetime.datetime.now()
        dist_ckpt_folder = nemo_dir / "model_weights"

        if dist_ckpt_folder.exists():
            weights_dict, llm_config, tokenizer = convert_dist_checkpoint(unpacked_checkpoint_dir, args)
        else:
            weights_dict, llm_config, tokenizer = convert_checkpoint(unpacked_checkpoint_dir, args)
        LOGGER.info("Spent %s (h:m:s) to convert the model", datetime.datetime.now() - start_time)

        if save_nemo_model_config:
            shutil.copyfile(unpacked_checkpoint_dir._checkpoints_dir / "model_config.yaml", args.out_dir / "model_config.yaml")

        return weights_dict, llm_config, tokenizer


def get_model_config(weights_dir: Path) -> GPT2Config:
    """Reads the GPT2Config from the decoded NEMO weights dir."""
    config = configparser.ConfigParser()
    config_path = weights_dir / "config.ini"
    assert os.path.isfile(config_path), f"{config_path} not present"
    config.read(config_path)
    config_dict = dict(config.items("gpt"))
    # Parse the config to dict.
    for k, v in config_dict.items():
        try:
            config_dict[k] = ast.literal_eval(v)
        except Exception:
            pass
    return GPT2Config(**config_dict)


def get_tokenzier(tokenizer_dir_or_path: Path) -> PreTrainedTokenizer:
    """Loads the tokenizer from the decoded NEMO weights dir."""
    if os.path.isdir(os.path.join(tokenizer_dir_or_path, "huggingface_tokenizer")):
        return AutoTokenizer.from_pretrained(os.path.join(tokenizer_dir_or_path, "huggingface_tokenizer"))

    model_path = (
        tokenizer_dir_or_path / "tokenizer.model"
        if tokenizer_dir_or_path.is_dir()
        else tokenizer_dir_or_path
    )
    tokenizer_config = {"library": "sentencepiece", "model": str(model_path)}
    return build_tokenizer(tokenizer_config)


def nemo_to_model_config(
    in_file: str, 
    decoder_type: str, 
    nemo_export_dir: typing.Union[str, Path], 
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    save_nemo_model_config: bool = False,
) -> Tuple[List[ModelConfig], PreTrainedTokenizer]:
    """Converts the NEMO file and construct the `ModelConfig` before tensorrt_llm deployment."""
    dtype_str = dtype

    weights_dict, llm_model_config, tokenizer = _nemo_decode(
        in_file=in_file,
        out_dir=nemo_export_dir,
        tensor_parallelism=tensor_parallel_size,
        processes=1,
        storage_type=dtype_str,
        load_checkpoints_on_gpu=False,
        decoder_type=decoder_type,
        save_nemo_model_config=save_nemo_model_config,
    )

    world_size = tensor_parallel_size*pipeline_parallel_size
    model_config_template = ModelConfig()
    model_config_template.dtype = dtype_str

    str_dtype_to_trt(dtype_str)

    model_configs = []
    for i in range(world_size):
        
        model_configs.append(copy.deepcopy(model_config_template))

        model_configs[i].vocab_embedding = EmbeddingConfig(
            weight=get_tensor_from_dict(weights_dict, "wte")
        )

        model_configs[i].final_layernorm = LayernormConfig(
            weight=get_tensor_from_dict(weights_dict, "final_layernorm.weight"),
            bias=get_tensor_from_dict(weights_dict, "final_layernorm.bias"),
        )
        model_configs[i].final_layernorm.layernorm_type = (
            LAYERNORM_RMS if isinstance(llm_model_config, LlamaConfig) else LAYERNORM_DEFAULT
        )
        model_configs[i].mapping = tensorrt_llm.Mapping(
            world_size=world_size, 
            rank=i, 
            tp_size=tensor_parallel_size, 
            pp_size=pipeline_parallel_size)

    for i in range(llm_model_config.n_layer):
        for j in range(world_size):
            model_configs[j].layers.append(
                DecoderLayerConfig.from_nemo(
                    weights_dict=weights_dict,
                    llm_config=llm_model_config,
                    decoder_type=decoder_type,
                    layer_id=i,
                    rank=model_configs[j].mapping.tp_rank,
                    is_mcore=llm_model_config.is_mcore,
                )
            )

    lm_head_weight = get_tensor_from_dict(weights_dict, "lm_head.weight")

    if model_configs[0].vocab_size_padded != model_configs[0].vocab_size:
        pad_width = model_configs[0].vocab_size_padded - model_configs[0].vocab_size
        lm_head_weight = np.pad(
            lm_head_weight, ((0, pad_width), (0, 0)), "constant", constant_values=0
        )

    for i in range(world_size):
        model_configs[i].lm_head = LinearConfig(linear_type=LINEAR_COLUMN)
        model_configs[i].lm_head.weight = np.ascontiguousarray(
            split(lm_head_weight, model_configs[i].mapping.tp_size, model_configs[i].mapping.tp_rank)
        )

    return model_configs, tokenizer
