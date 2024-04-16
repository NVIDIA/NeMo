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


import argparse
import copy
import csv
import datetime
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import tensorrt_llm
from tensorrt_llm import str_dtype_to_trt
from transformers import AutoTokenizer, LlamaConfig, PretrainedConfig, PreTrainedTokenizer

from nemo.export.trt_llm.model_config import (
    LAYERNORM_DEFAULT,
    LAYERNORM_RMS,
    LINEAR_COLUMN,
    DecoderLayerConfig,
    EmbeddingConfig,
    LayernormConfig,
    LinearConfig,
    ModelConfig,
)
from nemo.export.trt_llm.nemo.nemo import UnpackedNemoCheckpointDir, unpack_nemo_ckpt
from nemo.export.trt_llm.nemo.nemo_ckpt_convert import build_tokenizer, convert_dist_checkpoint, convert_nemo_model
from nemo.export.trt_llm.tensor_utils import get_tensor_from_dict, get_tensor_parallel_group, split

LOGGER = logging.getLogger("NeMo")


def _nemo_llm_decode(
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
            LOGGER.info("Spent %s (h:m:s) to unpack NeMo archive", datetime.datetime.now() - start_time)

        unpacked_checkpoint_dir = UnpackedNemoCheckpointDir(
            nemo_dir, load_checkpoints_to_cpu=not args.load_checkpoints_on_gpu
        )

        start_time = datetime.datetime.now()
        dist_ckpt_folder = nemo_dir / "model_weights"

        if dist_ckpt_folder.exists():
            weights_dict, llm_config, tokenizer = convert_dist_checkpoint(unpacked_checkpoint_dir, args)
        else:
            raise Exception(
                "Not a supported nemo file format. " "Only distributed mcore nemo checkpoints are support."
            )

        LOGGER.info("Spent %s (h:m:s) to convert the model", datetime.datetime.now() - start_time)

        if save_nemo_model_config:
            shutil.copyfile(
                unpacked_checkpoint_dir._checkpoints_dir / "model_config.yaml", args.out_dir / "model_config.yaml"
            )

        return weights_dict, llm_config, tokenizer


def get_tokenzier(tokenizer_dir_or_path: Path) -> PreTrainedTokenizer:
    """Loads the tokenizer from the decoded NEMO weights dir."""
    if os.path.isdir(os.path.join(tokenizer_dir_or_path, "huggingface_tokenizer")):
        return AutoTokenizer.from_pretrained(os.path.join(tokenizer_dir_or_path, "huggingface_tokenizer"))

    model_path = tokenizer_dir_or_path / "tokenizer.model" if tokenizer_dir_or_path.is_dir() else tokenizer_dir_or_path
    tokenizer_config = {"library": "sentencepiece", "model": str(model_path)}
    return build_tokenizer(tokenizer_config)


def nemo_llm_to_model_config(
    in_file: str,
    decoder_type: str,
    nemo_export_dir: Union[str, Path],
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    save_nemo_model_config: bool = False,
) -> Tuple[List[ModelConfig], PreTrainedTokenizer]:
    """Converts the NEMO file and construct the `ModelConfig` before tensorrt_llm deployment."""
    dtype_str = dtype

    weights_dict, llm_model_config, tokenizer = _nemo_llm_decode(
        in_file=in_file,
        out_dir=nemo_export_dir,
        tensor_parallelism=tensor_parallel_size,
        processes=1,
        storage_type=dtype_str,
        load_checkpoints_on_gpu=False,
        decoder_type=decoder_type,
        save_nemo_model_config=save_nemo_model_config,
    )

    world_size = tensor_parallel_size * pipeline_parallel_size
    model_config_template = ModelConfig()
    model_config_template.dtype = dtype_str

    str_dtype_to_trt(dtype_str)

    model_configs = []
    for i in range(world_size):

        model_configs.append(copy.deepcopy(model_config_template))

        model_configs[i].vocab_embedding = EmbeddingConfig(weight=get_tensor_from_dict(weights_dict, "wte"))

        model_configs[i].positional_embedding = EmbeddingConfig(weight=get_tensor_from_dict(weights_dict, "wpe"))

        model_configs[i].final_layernorm = LayernormConfig(
            weight=get_tensor_from_dict(weights_dict, "final_layernorm.weight"),
            bias=get_tensor_from_dict(weights_dict, "final_layernorm.bias"),
        )
        model_configs[i].final_layernorm.layernorm_type = (
            LAYERNORM_RMS if isinstance(llm_model_config, LlamaConfig) else LAYERNORM_DEFAULT
        )
        model_configs[i].mapping = tensorrt_llm.Mapping(
            world_size=world_size, rank=i, tp_size=tensor_parallel_size, pp_size=pipeline_parallel_size
        )

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
        lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)), "constant", constant_values=0)

    for i in range(world_size):
        model_configs[i].lm_head = LinearConfig(linear_type=LINEAR_COLUMN)
        model_configs[i].lm_head.weight = np.ascontiguousarray(
            split(lm_head_weight, model_configs[i].mapping.tp_size, model_configs[i].mapping.tp_rank)
        )

    return model_configs, tokenizer


def to_word_list_format(word_dict: List[List[str]], tokenizer=None):
    '''
    format of word_dict
        len(word_dict) should be same to batch_size
        word_dict[i] means the words for batch i
        len(word_dict[i]) must be 1, which means it only contains 1 string
        This string can contains several sentences and split by ",".
        For example, if word_dict[2] = " I am happy, I am sad", then this function will return
        the ids for two short sentences " I am happy" and " I am sad".
    '''
    assert tokenizer is not None, "need to set tokenizer"

    flat_ids = []
    offsets = []
    # We use a similar trick as in NeMo to deal with the fact that the encoding of a single word
    # can't always be trusted. See
    #   https://github.com/NVIDIA/NeMo/blob/bb575b72fd0be51ae10cc77d9f89ddb9e9d3b96d/nemo/collections/nlp/modules/common/text_generation_strategy.py#L229
    ids_ref = tokenizer.encode("<extra_id_1>")
    for word_dict_item in word_dict:
        item_flat_ids = []
        item_offsets = []

        if isinstance(word_dict_item[0], bytes):
            word_dict_item = [word_dict_item[0].decode()]

        words = list(csv.reader(word_dict_item))[0]
        for word in words:
            ids = tokenizer.encode(f"<extra_id_1>{word}")
            if ids[0 : len(ids_ref)] == ids_ref:
                # It worked! We can obtain the token(s) associated to `word` by stripping the prefix tokens.
                ids = ids[len(ids_ref) :]
            else:
                # Unfortunately the prefix was merged with `word`. We could try with a different prefix, but
                # for now we just use the basic encoding since this should be a very rare edge case.
                ids = tokenizer.encode(word)
                logging.warning(f"The encoding of word '{word}' into tokens {ids} might be incorrect")

            if len(ids) == 0:
                continue

            item_flat_ids += ids
            item_offsets.append(len(ids))

        flat_ids.append(np.array(item_flat_ids))
        offsets.append(np.cumsum(np.array(item_offsets)))

    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))


def nemo_llm_model_to_model_config(
    nemo_model, 
    tokenizer,
    nemo_model_config, 
    reshard_model,
    mapping
) -> Tuple[PretrainedConfig, dict]:
    """Converts the NEMO model object and construct the `ModelConfig` before tensorrt_llm deployment."""
    from megatron.core import parallel_state
    from tensorrt_llm.models.modeling_utils import PretrainedConfig
    from tensorrt_llm import Mapping

    weights_dict = convert_nemo_model(
        nemo_model=nemo_model, 
        nemo_model_config=nemo_model_config,
        tokenizer_vocab_size=tokenizer.vocab_size,
        reshard_model=reshard_model)

    # print(f"{torch.cuda.current_device()} {weights_dict.keys()}")
    # rank = torch.cuda.current_device() % 4
    # import safetensors
    # ckpt_dir = '/lustre/fsw/coreai_dlalgo_llm/jiemingz/tekit/examples/llama/2tllm_checkpoint_pp4_bf16'
    # model_path = os.path.join(ckpt_dir, f'rank{rank}.safetensors')
    # ref_weights = {}
    # with safetensors.safe_open(model_path, framework='pt', device='cpu') as f:
    #     for key in f.keys():
    #         ref_weights[key] = f.get_tensor(key)

    # for k in ref_weights.keys():
    #     v = ref_weights[k]
    #     v2 = weights_dict[k]
    #     if not torch.equal(v,v2):
    #         print(f"NE {rank} {k} {torch.sum(v)} {torch.sum(v2)}")
    # if torch.cuda.current_device() == 0:
    #     import pdb
    #     pdb.set_trace()
    # torch.distributed.barrier()

    if isinstance(nemo_model, list):
        torch_dtype = next(iter(nemo_model[0].state_dict().values())).dtype
    else:
        torch_dtype = next(iter(nemo_model.state_dict().values())).dtype

    str_dtype = trt_dtype_to_str(np_dtype_to_trt(torch_dtype_to_np(torch_dtype)))
    model_config = PretrainedConfig(
        architecture='LlamaForCausalLM',
        dtype=str_dtype,
        logits_dtype='float32',
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=nemo_model_config.get('max_position_embeddings'),
        hidden_size=nemo_model_config.get('hidden_size'),
        num_hidden_layers=nemo_model_config.get('num_layers'),
        num_attention_heads=nemo_model_config.get('num_attention_heads'),
        num_key_value_heads=nemo_model_config.get('num_query_groups'),
        hidden_act='silu',
        intermediate_size=nemo_model_config.get('ffn_hidden_size'),
        norm_epsilon=nemo_model_config.get('layernorm_epsilon'),
        position_embedding_type="rope_gpt_neox",
        world_size=mapping.world_size,
        tp_size=mapping.tp_size,
        pp_size=mapping.pp_size,
        quantization = {
            'quant_algo': None, 
            'kv_cache_quant_algo': None,
            'group_size': 128, 
            'has_zero_point': False, 
            'pre_quant_scale': False, 
            'exclude_modules': None}, 
        kv_dtype=str_dtype, 
        rotary_scaling=None,
        moe_normalization_mode=None, 
        rotary_base=10000.0, 
        moe_num_experts=0, 
        moe_top_k=0, 
        moe_tp_mode=2, 
        attn_bias=False, 
        disable_weight_only_quant_plugin=False, 
        mlp_bias=False
    )
    model_config.mapping = mapping
    return model_config, weights_dict
