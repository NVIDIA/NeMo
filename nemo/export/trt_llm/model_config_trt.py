# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The API convert the model_config format to tensorrt_llm."""

import os
import shutil
from pathlib import Path
from typing import List, Union

import psutil

from .model_config import ModelConfig
from .tensorrt_llm_model import LMHeadModelBuilder


def model_config_to_tensorrt_llm(
    model_configs: List[ModelConfig],
    engine_dir: Union[str, Path],
    gpus: int = 1,
    max_input_len: int = 200,
    max_output_len: int = 200,
    max_batch_size: int = 1,
    max_beam_width: int = 1,
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
    """
    engine_dir = Path(engine_dir)
    if os.path.exists(engine_dir):
        shutil.rmtree(engine_dir)

    print(
        "Before engine building, CPU RAM Used (GB):"
        f" {psutil.Process().memory_info().rss / 1024 / 1024 / 1024}"
    )
    for rank in range(gpus):
        builder = LMHeadModelBuilder(model_configs[rank])
        builder.build(
            output_dir=engine_dir,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            parallel_build=False,
        )
        print(
            f"After Engine building rank {rank}, CPU RAM Used (GB):"
            f" {psutil.Process().memory_info().rss / 1024 / 1024 / 1024}"
        )
