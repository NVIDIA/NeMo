import argparse
import configparser
import datetime
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import torch.multiprocessing as mp
from transformers import GPT2Config, PreTrainedTokenizer

from .nemo.nemo import UnpackedNemoCheckpointDir, unpack_nemo_ckpt
from .nemo.nemo_ckpt_convert import build_tokenizer, convert_checkpoint
from .tensorrt_llm_model import LMHeadModelBuilder

LOGGER = logging.getLogger(__name__)


def nemo_decode(
    in_file: str,
    out_dir: str,
    tensor_parallelism: int = 1,
    processes: int = 1,
    storage_type: str = "fp16",
    load_checkpoints_on_gpu: bool = False,
) -> Tuple[Path, GPT2Config, PreTrainedTokenizer]:
    """Decode the NEMO file and save the weights to out_dir."""
    args = argparse.Namespace()
    args.in_file = in_file
    args.out_dir = out_dir
    args.tensor_parallelism = tensor_parallelism
    args.processes = processes
    args.storage_type = storage_type
    args.load_checkpoints_on_gpu = load_checkpoints_on_gpu

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
        weights_dir, gpt_model_config, tokenizer = convert_checkpoint(unpacked_checkpoint_dir, args)
        LOGGER.info("Spent %s (h:m:s) to convert the model", datetime.datetime.now() - start_time)

        return weights_dir, gpt_model_config, tokenizer


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


def get_tokenzier(weights_dir: Path) -> PreTrainedTokenizer:
    """Loads the tokenizer from the decoded NEMO weights dir."""
    tokenizer_config = {"library": "sentencepiece", "model": str(weights_dir + "/tokenizer.model")}
    return build_tokenizer(tokenizer_config)


def _nemo_to_tensorrt_llm_impl(
    rank: int,
    weights_dir: Path,
    model_config: GPT2Config,
    engine_dir: str,
    gpus: int = 1,
    max_input_len=200,
    max_output_len=200,
    max_batch_size=1,
    max_beam_width=1,
    parallel_build=False,
):
    """The implmenetation of nemo_to_tensorrt_llm for a single rank."""

    builder = LMHeadModelBuilder(rank=rank, tensor_parallel=gpus)
    builder.load_nemo(weights_dir, model_config)

    builder.build(
        output_dir=engine_dir,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        max_batch_size=max_batch_size,
        max_beam_width=max_beam_width,
        parallel_build=parallel_build,
    )


def nemo_to_tensorrt_llm(
    weights_dir: Path,
    model_config: GPT2Config,
    engine_dir: str,
    gpus: int = 1,
    max_input_len=200,
    max_output_len=200,
    max_batch_size=1,
    max_beam_width=1,
    parallel_build=False,
):
    """The API to convert a nemo model to tensorrt_llm.

    gpus: the number of inference gpus for multi gpu inferencing.
    parallel_build: whether to build the multi gpu inference engine.
      Parallel build reduces the build time but increase the system memory load.
    """
    if gpus == 1:
        _nemo_to_tensorrt_llm_impl(
            0,
            weights_dir,
            model_config,
            engine_dir,
            gpus=1,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
        )
    elif parallel_build:
        mp.spawn(
            _nemo_to_tensorrt_llm_impl,
            nprocs=gpus,
            args=(
                weights_dir,
                model_config,
                engine_dir,
                gpus,
                max_input_len,
                max_output_len,
                max_batch_size,
                max_beam_width,
                parallel_build,
            ),
        )
    else:
        for rank in range(gpus):
            _nemo_to_tensorrt_llm_impl(
                rank,
                weights_dir,
                model_config,
                engine_dir,
                gpus=gpus,
                max_input_len=max_input_len,
                max_output_len=max_output_len,
                max_batch_size=max_batch_size,
                max_beam_width=max_beam_width,
                parallel_build=parallel_build,
            )
