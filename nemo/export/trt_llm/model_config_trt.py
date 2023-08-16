# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import os
import shutil
from typing import List

from .model_config import (
    ModelConfig,
)
from .tensorrt_llm_model import LMHeadModelBuilder


def model_config_to_tensorrt_llm(
    model_configs: List[ModelConfig],
    engine_dir: str,
    gpus: int = 1,
    max_input_len: int = 200,
    max_output_len: int = 200,
    max_batch_size: int = 1,
    max_beam_width: int = 1,
    max_prompt_embedding_table_size: int = 0,
):
    """The API to convert a torch or huggingface model represented as ModelConfig to tensorrt_llm.

    Args:
        model_configs: The list of ModelConfig converted, 1 for each GPU.
        engine_dir: The target output directory to save the built tensorrt_llm engines.
        gpus: the number of inference gpus for multi gpu inferencing.
        max_input_len: The max input sequence length.
        max_output_len: The max output sequence length.
        max_batch_size: The max batch size.
        max_beam_width: The max beam search width.
        max_prompt_embedding_table_size: The max prompt embedding table size.
    """
    if os.path.exists(engine_dir):
        shutil.rmtree(engine_dir)

    for rank in range(gpus):
        builder = LMHeadModelBuilder(model_configs[rank])
        builder.build(
            output_dir=engine_dir,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            parallel_build=False,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        )
