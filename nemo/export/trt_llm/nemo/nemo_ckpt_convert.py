# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Referrence impl in tensorrt_llm: examples/gpt/nemo_ckpt_convert.py."""

import configparser
import logging
import multiprocessing
import os
import shutil
import typing
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from tqdm import tqdm
from transformers import GPT2Tokenizer, LlamaConfig, T5Tokenizer

from .convert import (
    cpu_map_location,
    gpu_map_location,
    split_and_save_weight,
)
from .nemo import UnpackedNemoCheckpointDir, extract_layers_with_prefix, nemo_to_llm_config

LOGGER = logging.getLogger(__name__)

base_layer_names = {
    "position_embedding": "model.language_model.embedding.position_embeddings.weight",
    "word_embedding": "model.language_model.embedding.word_embeddings.weight",
    "output_layer": "model.language_model.output_layer.weight",
}

mcore_layer_names = {
    "position_embedding": "model.embedding.position_embeddings.weight",
    "word_embedding": "model.embedding.word_embeddings.weight",
    "output_layer": "model.output_layer.weight",
}


def get_layer_name(layer_type: str, is_mcore: bool):
    layer_dict = mcore_layer_names if is_mcore else base_layer_names
    if layer_type in layer_dict:
        return layer_dict[layer_type]
    else:
        raise ValueError(f"Unknown layer type {layer_type}")


def rename_key(old_key: str, pp_rank: int, num_layers: int, pp_size: int):
    new_key = old_key

    if "layers." in old_key:
        split_key = old_key.split(".")
        split_key[1] = str(int(split_key[1]) + pp_rank * num_layers // pp_size)
        new_key = ".".join(split_key)

        if "self_attention" in new_key:
            new_key = new_key.replace("self_attention", "attention")
    return new_key


@torch.no_grad()
def convert_checkpoint(unpacked_checkpoints_dir: UnpackedNemoCheckpointDir, args):
    nemo_model_config = unpacked_checkpoints_dir.model_config

    checkpoints_paths = unpacked_checkpoints_dir.get_checkpoints_paths(
        nemo_model_config.get("tensor_model_parallel_size", 1),
        nemo_model_config.get("pipeline_model_parallel_size", 1),
    )

    # if checkpoints files could be found - start preparing output dir
    out_dir = create_out_dir(args)

    map_location_fn = gpu_map_location if args.load_checkpoints_on_gpu else cpu_map_location
    storage_type = str_dtype_to_torch(args.storage_type)
    is_mcore = nemo_model_config.get("mcore_gpt", False)

    # load position_embedding from rank 0
    model_00 = torch.load(checkpoints_paths[0][0], map_location=map_location_fn)
    model_00 = model_00.get("state_dict", model_00)

    has_position_embedding = get_layer_name("position_embedding", is_mcore) in model_00
    has_lm_head = get_layer_name("output_layer", is_mcore) in model_00

    num_layers = nemo_model_config["num_layers"]
    training_tp_size = nemo_model_config.get("tensor_model_parallel_size", 1)
    training_pp_size = nemo_model_config.get("pipeline_model_parallel_size", 1)
    inference_tp_size = args.tensor_parallelism
    num_kv_heads = nemo_model_config.get("num_query_groups", 0)
    multi_query_mode = nemo_model_config.get("multi_query_mode", False)
    num_attention_heads = nemo_model_config["num_attention_heads"]
    if num_kv_heads == 0:
        if multi_query_mode:
            num_kv_heads = 1
        else:
            num_kv_heads = num_attention_heads

    export_config = {
        "apply_layernorm_1p": nemo_model_config.get("normalization", "") == "layernorm1p",
        "tp_size": training_tp_size,
        "split_gated_activation": "swiglu" in nemo_model_config.get("activation", "gelu") and (
            args.decoder_type == "gptnext" or is_mcore
        ),
        "num_attention_heads": num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "use_attention_nemo_shape": True,
        "transpose_weights": True,
    }

    # merge_factor: how many TP training nodes are merged into an inference TP node
    # split_factor: in how many parts a TP training node is split
    gcd = np.gcd(training_tp_size, inference_tp_size)
    merge_factor = training_tp_size // gcd
    split_factor = inference_tp_size // gcd

    model_level_weights = defaultdict(list)

    def handle_model_level_weights(model, tp_idx: int, pp_idx: int):
        if tp_idx == 0 and pp_idx == 0:
            if has_position_embedding:
                val = model[get_layer_name("position_embedding", is_mcore)]
                # not weight, do not need to transpose
                val = torch_to_numpy(val.to(storage_type).cpu())
                # AMMO modification
                # val.tofile(out_dir / "model.wpe.bin")
                model_level_weights["model.wpe.bin"].append(val)
        if pp_idx == 0:
            val = model.get("state_dict", model)[get_layer_name("word_embedding", is_mcore)]
            val = torch_to_numpy(val.to(storage_type).cpu())
            model_level_weights["model.wte.bin"].append(val)
        if has_lm_head and pp_idx == training_pp_size - 1:
            val = model.get("state_dict", model)[get_layer_name("output_layer", is_mcore)]
            val = torch_to_numpy(val.to(storage_type).cpu())
            model_level_weights["model.lm_head.weight.bin"].append(val)

    # AMMO modification
    weights_dict = {}
    for tp_rank in range(training_tp_size // merge_factor):
        for pp_rank in range(training_pp_size):
            models = []
            for k in range(merge_factor):
                rank_weights = checkpoints_paths[tp_rank * merge_factor + k][pp_rank]
                model = torch.load(rank_weights, map_location=map_location_fn)
                handle_model_level_weights(model, tp_rank * merge_factor + k, pp_rank)
                prefix = "model.decoder." if is_mcore else "model.language_model.encoder."
                layers = extract_layers_with_prefix(model, prefix)
                models.append(layers)

            starmap_args = []
            for key in models[0].keys():
                # Skipping the extra state as it is not a part of the model state dict
                if "_extra_state" not in key:
                    starmap_args.append(
                        (
                            tp_rank,
                            out_dir,
                            split_factor,
                            rename_key(key, pp_rank, num_layers, training_pp_size),
                            [model[key] for model in models],
                            storage_type,
                            None,
                            export_config,
                        )
                    )
            starmap_args = tqdm(starmap_args, desc="saving weights")

            if args.processes > 1:
                with multiprocessing.Pool(args.processes) as pool:
                    # AMMO modification
                    weights_dicts = pool.starmap(split_and_save_weight, starmap_args)
                    weights_dict_local = {k: v for d in weights_dicts for k, v in d.items()}
            else:
                # simpler for debug situations
                for starmap_arg in starmap_args:
                    # AMMO modification
                    weights_dict_local = split_and_save_weight(*starmap_arg)
            # AMMO modification
            weights_dict.update(weights_dict_local)

    for key, values in model_level_weights.items():
        model_level_weights[key] = np.concatenate(values, axis=0)
        # AMMO modification
        weights_dict[key] = model_level_weights[key]
    vocab_size = model_level_weights["model.wte.bin"].shape[0]

    tokenizer_config = update_tokenizer_paths(
        nemo_model_config["tokenizer"], unpacked_checkpoints_dir
    )
    copy_tokenizer_files(tokenizer_config, out_dir)
    # AMMO modification.
    tokenizer_config["model"] = os.path.join(out_dir, "tokenizer.model")
    tokenizer = build_tokenizer(tokenizer_config)
    llm_config = nemo_to_llm_config(
        nemo_model_config,
        vocab_size,
        tokenizer.eos_token_id,
        tokenizer.bos_token_id,
        args.decoder_type,
    )

    llm_config.is_mcore = is_mcore

    config = configparser.ConfigParser()
    model_name = "llama" if isinstance(llm_config, LlamaConfig) else "gpt"
    config[model_name] = {k: str(v) for k, v in vars(llm_config).items()}
    config[model_name]["storage_dtype"] = args.storage_type
    config_path = out_dir / "config.ini"
    with config_path.open("w") as config_file:
        config.write(config_file)

    # AMMO modification.
    return weights_dict, llm_config, tokenizer


def create_out_dir(args):
    # AMMO modification.
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    return out_dir


def update_tokenizer_paths(tokenizer_config: typing.Dict, unpacked_checkpoints_dir):
    def _update_config_entry(key, file_pattern):
        old_path = tokenizer_config[key]
        if old_path is None:
            return
        old_path = Path(old_path)
        new_path = unpacked_checkpoints_dir.get_tokenizer_file_path("tokenizer", key, file_pattern)
        if new_path:
            LOGGER.debug(f"Update tokenizer {key} {old_path} -> {new_path}")
            tokenizer_config[key] = new_path.as_posix()
        elif not old_path.exists():
            LOGGER.warning(f"Tokenizer {key}'s path {old_path} does not exists: set it to None")
            tokenizer_config[key] = None

    _update_config_entry("model", "*.model")
    _update_config_entry("vocab_file", "*vocab*")
    _update_config_entry("merge_file", "*merge*.txt")

    return tokenizer_config


def copy_tokenizer_files(config, out_dir):
    basenames = {
        "model": "tokenizer",
        "vocab_file": "vocab",
        "merge_file": "merges",
    }

    for key in basenames.keys():
        if config[key] is None:
            continue
        path = Path(config[key])
        if not path.exists():
            LOGGER.debug(f"Tokenizer {key}: {path} file not found")
            continue

        dst_path = out_dir / f"{basenames[key]}{path.suffix}"
        LOGGER.debug(f"Copy tokenizer {key}: {path}->{dst_path}")
        shutil.copy(path.as_posix(), dst_path.as_posix())


def build_tokenizer(tokenizer_config: typing.Dict):
    if tokenizer_config["library"] == "sentencepiece":
        # AMMO modification.
        # Turn off legacy model by default: See https://github.com/huggingface/transformers/pull/24622
        tokenizer = T5Tokenizer(tokenizer_config["model"], extra_ids=0, legacy=False)
    elif "GPT2" in tokenizer_config["type"]:
        tokenizer = GPT2Tokenizer(tokenizer_config["vocab_file"], tokenizer_config["merge_file"])
    else:
        raise ValueError(f'Tokenizer type {tokenizer_config["library"]} not handled')

    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": "<s>"})
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})

    return tokenizer
