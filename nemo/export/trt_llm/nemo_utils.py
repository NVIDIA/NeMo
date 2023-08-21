import argparse
import configparser
import copy
import datetime
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tensorrt_llm._utils import str_dtype_to_trt
from transformers import GPT2Config, LlamaConfig, PretrainedConfig, PreTrainedTokenizer

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
from .nemo.nemo_ckpt_convert import build_tokenizer, convert_checkpoint
from .tensor_utils import get_tensor_from_file, split

LOGGER = logging.getLogger(__name__)


def _nemo_decode(
    in_file: str,
    out_dir: str,
    tensor_parallelism: int = 1,
    processes: int = 1,
    storage_type: str = "bfloat16",
    load_checkpoints_on_gpu: bool = False,
    decoder_type: str = "gptnext",
) -> Tuple[PretrainedConfig, PreTrainedTokenizer]:
    """Decodes the NEMO file and save the weights to out_dir."""
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
        llm_config, tokenizer = convert_checkpoint(unpacked_checkpoint_dir, args)
        LOGGER.info("Spent %s (h:m:s) to convert the model", datetime.datetime.now() - start_time)

        return llm_config, tokenizer


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
            config_dict[k] = eval(v)
        except Exception:
            pass
    return GPT2Config(**config_dict)


def get_tokenzier(tokenizer_dir_or_path: Path) -> PreTrainedTokenizer:
    """Loads the tokenizer from the decoded NEMO weights dir."""
    model_path = tokenizer_dir_or_path / "tokenizer.model" if tokenizer_dir_or_path.is_dir() else tokenizer_dir_or_path
    tokenizer_config = {"library": "sentencepiece", "model": str(model_path)}
    return build_tokenizer(tokenizer_config)


def nemo_to_model_config(
    in_file: str, decoder_type: str, gpus: int = 1, nemo_export_dir: str = "/tmp/nemo"
) -> Tuple[List[ModelConfig], PreTrainedTokenizer]:
    """Converts the NEMO file and construct the `ModelConfig` before tensorrt_llm deployment."""
    dtype_str = "bfloat16"

    if os.path.exists(nemo_export_dir):
        shutil.rmtree(nemo_export_dir)

    llm_model_config, tokenizer = _nemo_decode(
        in_file=in_file,
        out_dir=nemo_export_dir,
        tensor_parallelism=gpus,
        processes=1,
        storage_type=dtype_str,
        load_checkpoints_on_gpu=False,
        decoder_type=decoder_type,
    )

    weights_dir = Path(nemo_export_dir)

    model_config_template = ModelConfig()
    model_config_template.dtype = dtype_str

    model_config_template.tensor_parallel = gpus

    dtype = str_dtype_to_trt(dtype_str)

    vocab_size = llm_model_config.vocab_size
    hidden_size = llm_model_config.n_embd
    model_config_template.vocab_embedding = EmbeddingConfig(
        weight=get_tensor_from_file(weights_dir, "wte", shape=[vocab_size, hidden_size], dtype=dtype)
    )

    model_config_template.final_layernorm = LayernormConfig(
        weight=get_tensor_from_file(weights_dir, "final_layernorm.weight", dtype=dtype),
        bias=get_tensor_from_file(weights_dir, "final_layernorm.bias", dtype=dtype),
    )

    model_config_template.final_layernorm.layernorm_type = (
        LAYERNORM_RMS if isinstance(llm_model_config, LlamaConfig) else LAYERNORM_DEFAULT
    )

    model_configs = []
    for i in range(gpus):
        model_configs.append(copy.deepcopy(model_config_template))
        model_configs[i].rank = i

    for i in range(llm_model_config.n_layer):
        for j in range(gpus):
            model_configs[j].layers.append(
                DecoderLayerConfig.from_nemo(
                    weights_dir=weights_dir,
                    llm_config=llm_model_config,
                    decoder_type=decoder_type,
                    layer_id=i,
                    rank=j,
                    tensor_parallel=gpus,
                    dtype=dtype,
                )
            )

    lm_head_weight = get_tensor_from_file(weights_dir, "lm_head.weight", shape=[vocab_size, hidden_size], dtype=dtype)

    if model_configs[0].vocab_size_padded != model_configs[0].vocab_size:
        pad_width = model_configs[0].vocab_size_padded - model_configs[0].vocab_size
        lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)), "constant", constant_values=0)

    for i in range(gpus):
        model_configs[i].lm_head = LinearConfig(linear_type=LINEAR_COLUMN)
        model_configs[i].lm_head.weight = np.ascontiguousarray(
            split(lm_head_weight, model_configs[i].tensor_parallel, model_configs[i].rank)
        )

    return model_configs, tokenizer
