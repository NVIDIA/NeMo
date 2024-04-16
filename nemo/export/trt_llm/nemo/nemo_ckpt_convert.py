# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import configparser
import logging
import math
import multiprocessing
import os
import shutil
import typing
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorstore  # This is important even though not used. Otherwise zarr raises error.
import torch
import zarr
from tensorrt_llm._utils import np_bfloat16, str_dtype_to_torch, torch_to_numpy
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Tokenizer, LlamaConfig

from nemo.export.trt_llm.nemo.convert import save_weight_torch, split_and_save_weight
from nemo.export.trt_llm.nemo.nemo import UnpackedNemoCheckpointDir, extract_layers_with_prefix, nemo_to_llm_config
from nemo.export.trt_llm.nemo.sentencepiece_tokenizer import SentencePieceTokenizer


LOGGER = logging.getLogger("NeMo")

layer_names = {
    "position_embedding": "embedding.position_embeddings.weight",
    "word_embedding": "embedding.word_embeddings.weight",
    "output_layer": "output_layer.weight",
    "final_layernorm.weight": "final_layernorm.weight",
    "final_layernorm.bias": "final_layernorm.bias",
}


def get_layer_name(layer_type: str, prefix: str):
    layer_dict = layer_names
    if layer_type in layer_dict:
        return prefix + layer_dict[layer_type]
    else:
        raise ValueError(f"Unknown layer type {layer_type}")


def get_layer_prefix(layer_names, is_mcore):
    transformer_layer_prefix = None

    for layer_name in layer_names:
        if 'self_attention' in layer_name:
            transformer_layer_prefix = layer_name.split('layers')[0]
            break
    assert transformer_layer_prefix is not None, "Cannot extract transformer layer prefix from {layer_name}"
    if is_mcore:
        model_prefix = transformer_layer_prefix.split('decoder')[0]
    else:
        model_prefix = transformer_layer_prefix.split('encoder')[0]
    assert model_prefix is not None, "Cannot extract model prefix from {layer_name}"

    return model_prefix, transformer_layer_prefix


def get_layer_index(split_key):
    index = 0
    for key in split_key:
        if key == "layers":
            return index + 1
        index += 1


def rename_key(old_key: str, pp_rank: int, num_layers: int, pp_size: int):
    new_key = old_key

    if "layers." in old_key:
        split_key = old_key.split(".")
        layer_index = get_layer_index(split_key)
        split_key[layer_index] = str(int(split_key[layer_index]) + pp_rank * num_layers // pp_size)
        new_key = ".".join(split_key)

        if "self_attention" in new_key:
            new_key = new_key.replace("self_attention", "attention")
        if "attention.linear_qkv.layer_norm_weight" in new_key:
            new_key = new_key.replace("attention.linear_qkv.layer_norm_weight", "input_layernorm.weight")
        if "mlp.linear_fc1.layer_norm_weight" in new_key:
            new_key = new_key.replace("mlp.linear_fc1.layer_norm_weight", "post_attention_layernorm.weight")

    return new_key


def rename_key_dist_ckpt(old_key: str, layer: int):
    new_key = old_key

    if "layers." in old_key:
        split_key = old_key.split(".")
        split_key.insert(1, str(layer))
        new_key = ".".join(split_key)

        if "self_attention" in new_key:
            new_key = new_key.replace("self_attention", "attention")
        if "attention.linear_qkv.layer_norm_weight" in new_key:
            new_key = new_key.replace("attention.linear_qkv.layer_norm_weight", "input_layernorm.weight")
        if "attention.linear_qkv.layer_norm_bias" in new_key:
            new_key = new_key.replace("attention.linear_qkv.layer_norm_bias", "input_layernorm.bias")
        if "mlp.linear_fc1.layer_norm_weight" in new_key:
            new_key = new_key.replace("mlp.linear_fc1.layer_norm_weight", "post_attention_layernorm.weight")
        if "mlp.linear_fc1.layer_norm_bias" in new_key:
            new_key = new_key.replace("mlp.linear_fc1.layer_norm_bias", "post_attention_layernorm.bias")

    return new_key


def load_sharded_metadata(checkpoint_dir: str, torch_tensor=True):
    checkpoint_dir = Path(checkpoint_dir)
    sharded_state_dict = {}
    for subdir in checkpoint_dir.iterdir():
        if not subdir.is_dir() or not (subdir / '.zarray').exists():
            continue
        key = subdir.name
        arr = zarr.open(str(subdir), 'r')
        if torch_tensor:
            # sharded_state_dict[key] = torch.from_numpy(arr[:].astype("float32")).to(dtype=torch.bfloat16)
            if arr.dtype.name == "bfloat16":
                sharded_state_dict[key] = torch.from_numpy(arr[:].view(np.int16)).view(torch.bfloat16)
            else:
                sharded_state_dict[key] = torch.from_numpy(arr[:])
        else:
            sharded_state_dict[key] = arr[:]

    return sharded_state_dict


@torch.no_grad()
def convert_dist_checkpoint(unpacked_checkpoints_dir: UnpackedNemoCheckpointDir, args):
    nemo_model_config = unpacked_checkpoints_dir.model_config
    checkpoints_path = unpacked_checkpoints_dir.checkpoints_dir / "model_weights"

    # if checkpoints files could be found - start preparing output dir
    out_dir = create_out_dir(args)

    storage_type = str_dtype_to_torch(args.storage_type)
    is_mcore = nemo_model_config.get("mcore_gpt", False)

    # load position_embedding from rank 0
    model = load_sharded_metadata(checkpoints_path)
    model_state_dict = model.get("state_dict", model)

    prefix, transformer_layer_prefix = get_layer_prefix(model_state_dict.keys(), is_mcore)

    has_position_embedding = get_layer_name("position_embedding", prefix) in model_state_dict
    has_lm_head = get_layer_name("output_layer", prefix) in model_state_dict
    share_embeddings_and_output = nemo_model_config.get("share_embeddings_and_output_weights", False)
    embedding_scaling = nemo_model_config.get("apply_embedding_scaling", False)
    hidden_size = nemo_model_config["hidden_size"]

    num_layers = nemo_model_config["num_layers"]
    training_tp_size = 1
    training_pp_size = 1
    inference_tp_size = args.tensor_parallelism
    num_kv_heads = nemo_model_config.get("num_query_groups", 0)
    multi_query_mode = nemo_model_config.get("multi_query_mode", False)
    num_attention_heads = nemo_model_config["num_attention_heads"]
    kv_channels = nemo_model_config.get("kv_channels", None)
    if num_kv_heads == 0:
        if multi_query_mode:
            num_kv_heads = 1
        else:
            num_kv_heads = num_attention_heads

    export_config = {
        "apply_layernorm_1p": nemo_model_config.get("normalization", "") == "layernorm1p",
        "tp_size": training_tp_size,
        "split_gated_activation": nemo_model_config.get("activation", "gelu")
        in ["swiglu", "geglu", "fast-swiglu", "fast-geglu"]
        and (args.decoder_type == "gptnext" or is_mcore),
        "num_attention_heads": num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "kv_channels": kv_channels,
        "use_attention_nemo_shape": True,
        "transpose_weights": True,
    }

    # split_factor: in how many parts a TP training node is split
    split_factor = inference_tp_size
    model_level_weights = defaultdict(list)

    def handle_model_level_weights(model, tp_idx: int, pp_idx: int):
        if tp_idx == 0 and pp_idx == 0:
            if has_position_embedding:
                val = model[get_layer_name("position_embedding", prefix)]
                val = torch_to_numpy(val.to(storage_type).cpu())
                model_level_weights["model.wpe.bin"].append(val)
        if pp_idx == 0:
            val = model.get("state_dict", model)[get_layer_name("word_embedding", prefix)]
            if embedding_scaling:
                val = val * float(math.sqrt(hidden_size))

            val = torch_to_numpy(val.to(storage_type).cpu())
            model_level_weights["model.wte.bin"].append(val)
            if share_embeddings_and_output:
                val = model.get("state_dict", model)[get_layer_name("word_embedding", prefix)]
                val = torch_to_numpy(val.to(storage_type).cpu())
                model_level_weights["model.lm_head.weight.bin"].append(val)
        if has_lm_head and pp_idx == training_pp_size - 1:
            val = model.get("state_dict", model)[get_layer_name("output_layer", prefix)]
            val = torch_to_numpy(val.to(storage_type).cpu())
            model_level_weights["model.lm_head.weight.bin"].append(val)

    weights_dict = {}

    tp_rank = 0

    handle_model_level_weights(model, 0, 0)
    model = extract_layers_with_prefix(model, transformer_layer_prefix)

    starmap_args = []
    for key, val in model.items():
        if "_extra_state" not in key:
            if len(val.size()) == 1:
                starmap_args.append(
                    (
                        tp_rank,
                        out_dir,
                        split_factor,
                        # Let's rename/map the key to the old layer name previously. You can try printing out
                        # the rename_key output of the old llama checkpoint and compare.
                        rename_key_dist_ckpt(key, 0),
                        # Since the state dict value has the full layers, let's select the ith layer weights/biases here.
                        [val],
                        storage_type,
                        None,
                        export_config,
                    )
                )
            else:
                for i in range(num_layers):
                    starmap_args.append(
                        (
                            tp_rank,
                            out_dir,
                            split_factor,
                            # Let's rename/map the key to the old layer name previously. You can try printing out
                            # the rename_key output of the old llama checkpoint and compare.
                            rename_key_dist_ckpt(key, i),
                            # Since the state dict value has the full layers, let's select the ith layer weights/biases here.
                            [val[i]],
                            storage_type,
                            None,
                            export_config,
                        )
                    )

    starmap_args = tqdm(starmap_args, desc="saving weights")

    if args.processes > 1:
        with multiprocessing.Pool(args.processes) as pool:
            weights_dicts = pool.starmap(split_and_save_weight, starmap_args)
            weights_dict_local = {k: v for d in weights_dicts for k, v in d.items()}
    else:
        # simpler for debug situations
        for starmap_arg in starmap_args:
            weights_dict_local = split_and_save_weight(*starmap_arg)

    weights_dict.update(weights_dict_local)

    for key, values in model_level_weights.items():
        model_level_weights[key] = np.concatenate(values, axis=0)

        weights_dict[key] = model_level_weights[key]
    vocab_size = model_level_weights["model.wte.bin"].shape[0]

    if nemo_model_config["tokenizer"].get("library", None) == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(
            nemo_model_config["tokenizer"]["type"], use_fast=nemo_model_config["tokenizer"].get("use_fast", False)
        )
    else:
        tokenizer_config = update_tokenizer_paths(nemo_model_config["tokenizer"], unpacked_checkpoints_dir)
        copy_tokenizer_files(tokenizer_config, out_dir)

        tokenizer_config["model"] = os.path.join(out_dir, "tokenizer.model")
        tokenizer = build_tokenizer(tokenizer_config)

    llm_config = nemo_to_llm_config(
        nemo_model_config, vocab_size, tokenizer.eos_token_id, tokenizer.bos_token_id, args.decoder_type,
    )

    llm_config.is_mcore = is_mcore

    config = configparser.ConfigParser()
    decoder_name_dict = {"llama": "llama", "falcon": "falcon"}
    model_name = decoder_name_dict[args.decoder_type] if args.decoder_type in decoder_name_dict else "gpt"

    config[model_name] = {k: str(v) for k, v in vars(llm_config).items()}
    config[model_name]["storage_dtype"] = args.storage_type
    config_path = out_dir / "config.ini"
    with config_path.open("w") as config_file:
        config.write(config_file)

    return weights_dict, llm_config, tokenizer


@torch.no_grad()
def convert_nemo_model(nemo_model, nemo_model_config,  tokenizer_vocab_size, reshard_model=False, cpu=True):
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.utils import VocabUtility

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_first_rank = parallel_state.get_pipeline_model_parallel_first_rank()
    pp_last_rank = parallel_state.get_pipeline_model_parallel_last_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
    if not vp_size: vp_size = 1

    num_layers = nemo_model_config["num_layers"]
    training_pp_size = nemo_model_config.get("pipeline_model_parallel_size", 1)
    num_kv_heads = nemo_model_config.get("num_query_groups", 0)
    multi_query_mode = nemo_model_config.get("multi_query_mode", False)
    num_attention_heads = nemo_model_config["num_attention_heads"]
    is_mcore = nemo_model_config.get("mcore_gpt", False)

    if vp_size > 1:
        state_dict = nemo_model[0].state_dict()
    else:
        state_dict = nemo_model.state_dict()
    storage_type = next(iter(state_dict.values())).dtype
    prefix, transformer_layer_prefix = get_layer_prefix(state_dict, is_mcore)

    if num_kv_heads == 0:
        num_kv_heads = 1 if multi_query_mode else num_attention_heads
    reshard_model = reshard_model and pp_size > 1
    weights_dict = persistent_weight_dict if cpu else {}

    export_config = {
        "apply_layernorm_1p": nemo_model_config.get("normalization", "") == "layernorm1p",
        "tp_size": tp_size,
        "split_gated_activation": "swiglu" in nemo_model_config.get("activation", "gelu"),
        "num_attention_heads": nemo_model_config["num_attention_heads"],
        "num_kv_heads": num_kv_heads,
        "transpose_weights": True,
        "num_layers": num_layers,
        "storage_type": storage_type,
        "move_to_cpu": cpu,
        "save_dict": weights_dict,
        "tp_rank": tp_rank
    }

    tl_params = {}
    model_level_params = {}
    starmap_args = []

    tic = time.time()

    layers_per_pp = num_layers // pp_size
    layers_per_chunk = layers_per_pp // vp_size

    if vp_size > 1: # consolidate params across model chunks
        for idx, model_chunk in enumerate(nemo_model):
            for key, val in model_chunk.state_dict().items():
                if '_extra_state' in key:
                    continue
                elif 'decoder.layers' in key:
                    key2 = rename_layer_num(key, get_layer_num(key) + idx*pp_size*layers_per_chunk)
                    tl_params[key2] = val
                else:
                    model_level_params[key] = val
    else:
        for key, val in nemo_model.state_dict().items():
            if '_extra_state' in key:
                continue
            elif 'decoder.layers' in key:
                tl_params[key] = val
            else:
                model_level_params[key] = val  

    if vp_size > 1 or reshard_model:
        # gather layers across pp ranks
        gathered_params = {}
        for key, val in tl_params.items():
            weight_list = [torch.zeros_like(val) for _ in range(pp_size)]
            torch.distributed.all_gather(weight_list, val, group=pp_group)
            for idx in range(pp_size):  
                layer_num = get_layer_num(key) + idx*layers_per_chunk
                key2 = rename_layer_num(key, layer_num)
                if not reshard_model: #Save only layers of 1 single PP stage
                    layers_start = layers_per_pp*pp_rank
                    layers_end = layers_per_pp*(pp_rank+1) -1
                    if layer_num >= layers_start and layer_num <= layers_end:
                        key2 = rename_layer_num(key, layer_num % layers_per_pp)
                        gathered_params[key2] = weight_list[idx]
                else:
                    gathered_params[key2] = weight_list[idx]
        tl_params = gathered_params

    toc = time.time()
    print(f"    PP Reshard save took {toc-tic}")

    # ----------------Convert layer level weights----------------  
    layer_params = extract_layers_with_prefix(tl_params, transformer_layer_prefix)
    layer_params = {
        k: v for k, v in layer_params.items() if k.startswith("layers.")
    }
    for key, val in layer_params.items():
        starmap_args.append(
            {
                "key": key,
                "val": val,
                "config": export_config,
            }
        )

    def broadcast_item(item, group, src_rank):
        item = [item]
        torch.distributed.broadcast_object_list(item, src_rank, group=group)
        return item[0]

    #broadcast a tensor across PP group and save it
    def broadcast_save_weight(
        src_key_or_tensor, dst_key, pp_src_idx, transpose_weights=False):

        if (not reshard_model) or (reshard_model and torch.distributed.get_rank() == pp_src_idx):
            if torch.is_tensor(src_key_or_tensor):
                tensor = src_key_or_tensor
            else:
                tensor = model_level_params[src_key_or_tensor]

        if reshard_model:
            if torch.distributed.get_rank() == pp_src_idx:
                shape = tensor.shape
            else:
                shape = [None]
            shape = broadcast_item(shape, pp_group, pp_src_idx)
            
            if torch.distributed.get_rank() != pp_src_idx:
                tensor = torch.zeros(shape, dtype=storage_type).cuda()
            torch.distributed.broadcast(tensor, pp_src_idx, group=pp_group)

        temp_config = dict(export_config)
        temp_config['transpose_weights'] = transpose_weights
        starmap_args.append(
                {
                    "key": dst_key,
                    "val": tensor,
                    "config": temp_config,
                }
            )

    # ----------------Convert Final Layernorm----------------  
    if torch.distributed.get_rank() == pp_last_rank or reshard_model:
        broadcast_save_weight(
            get_layer_name("final_layernorm.weight", transformer_layer_prefix), 
            "ln_f.weight", 
            pp_last_rank, 
            transpose_weights=True
        )

    has_final_layer_bias = get_layer_name("final_layernorm.bias", transformer_layer_prefix) in model_level_params
    if reshard_model:
        has_final_layer_bias = broadcast_item(has_final_layer_bias, pp_group, pp_last_rank)
    if has_final_layer_bias:
        broadcast_save_weight(
            get_layer_name("final_layernorm.bias", transformer_layer_prefix), 
            "ln_f.bias", 
            pp_last_rank, 
            transpose_weights=True
        )

    # ----------------Convert Embeddings----------------  
    def remove_vocab_padding(tensor):
        vocab_size_per_tp = tensor.shape[0]
        vocab_size_padded = vocab_size_per_tp*tp_size
        vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
        vocab_size_padded, tp_rank, tp_size)

        dim_size = list(tensor.size())
        dim_size[0] = vocab_size_padded

        gathered_tensor = torch.zeros(dim_size, dtype=tensor.dtype).cuda()
        gathered_tensor[vocab_start_index:vocab_end_index] = tensor
        torch.distributed.all_reduce(gathered_tensor, group=tp_group)
        return gathered_tensor[:tokenizer_vocab_size]

    if torch.distributed.get_rank() == pp_first_rank:
        world_embed = model_level_params[get_layer_name("word_embedding", prefix)]
        if tp_size > 1:
            world_embed = remove_vocab_padding(world_embed)
    else:
        world_embed = None

    if torch.distributed.get_rank() == pp_first_rank or reshard_model:
        broadcast_save_weight(
            world_embed, 
            "transformer.vocab_embedding.weight", 
            pp_first_rank, 
            transpose_weights=False, 
        )

    if torch.distributed.get_rank() == pp_last_rank:
        lm_head = model_level_params[get_layer_name("output_layer", prefix)]
        if tp_size > 1:
            lm_head = remove_vocab_padding(lm_head)

            vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                tokenizer_vocab_size, tp_rank, tp_size)
            lm_head = lm_head[vocab_start_index:vocab_end_index]
    else:
        lm_head = None

    if torch.distributed.get_rank() == pp_last_rank or reshard_model:
        broadcast_save_weight(
            lm_head, 
            "lm_head.weight", 
            pp_last_rank,
            transpose_weights=False, 
        )
    tic = time.time()
    for starmap_arg in tqdm(starmap_args, desc="saving weights"):
        save_weight_torch(**starmap_arg)
    toc = time.time()
    print(f"     weight save took {toc-tic}")
    return weights_dict



def create_out_dir(args):
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


def build_tokenizer(tokenizer):
    if isinstance(tokenizer, dict):
        tokenizer_config = tokenizer
        if tokenizer_config["library"] == "sentencepiece":
            return SentencePieceTokenizer(model_path=tokenizer_config["model"])
        elif "GPT2" in tokenizer_config["type"]:
            tokenizer = GPT2Tokenizer(tokenizer_config["vocab_file"], tokenizer_config["merge_file"])
        else:
            raise ValueError(f'Tokenizer type {tokenizer_config["library"]} not handled')

        if tokenizer.bos_token_id is None:
            tokenizer.add_special_tokens({"bos_token": "<s>"})
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"eos_token": "</s>"})
    else:
        try:
            # If NeMo tokenizer, monkey patch interface
            from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

            if isinstance(tokenizer, TokenizerSpec):

                def batch_encode_patch(self, ids):
                    if torch.is_tensor(ids):
                        ids = ids.cpu().numpy()
                    return self.ids_to_text(ids)

                tokenizer.bos_token_id = tokenizer.bos_id
                tokenizer.eos_token_id = tokenizer.eos_id
                tokenizer.encode = tokenizer.text_to_ids
                TokenizerSpec.batch_decode = batch_encode_patch
        except:
            raise TypeError(f'Unsupported tokenizer build input: {type(tokenizer)}')

    return tokenizer
