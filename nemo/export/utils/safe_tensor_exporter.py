# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import safetensors
import torch

from nemo.export.trt_llm.converter.model_converter import model_to_trtllm_ckpt
from nemo.export.trt_llm.nemo_ckpt_loader.nemo_file import load_nemo_model

LOGGER = logging.getLogger("NeMo")

np_bfloat16 = np.dtype('V2', metadata={"dtype": "bfloat16"})
np_float8 = np.dtype('V1', metadata={"dtype": "float8"})


def numpy_to_torch(x):
    """Convert numpy to torch"""
    if x.dtype == np_bfloat16:
        return torch.from_numpy(x.view(np.int16)).view(torch.bfloat16)
    elif x.dtype == np_float8:
        return torch.from_numpy(x.view(np.int8)).view(torch.float8_e4m3fn)
    else:
        return torch.from_numpy(x)


def convert_to_safe_tensors(
    nemo_checkpoint_path: str,
    model_dir: str,
    model_type: Optional[str] = None,
    delete_existing_files: bool = True,
    tensor_parallelism_size: int = 1,
    pipeline_parallelism_size: int = 1,
    gpus_per_node: int = None,
    use_parallel_embedding: bool = False,
    use_embedding_sharing: bool = False,
    dtype: str = "bfloat16",
):
    """Convert to safe tensor"""
    gpus_per_node = tensor_parallelism_size if gpus_per_node is None else gpus_per_node

    if Path(model_dir).exists():
        if delete_existing_files and len(os.listdir(model_dir)) > 0:
            for files in os.listdir(model_dir):
                path = os.path.join(model_dir, files)
                try:
                    shutil.rmtree(path)
                except OSError:
                    os.remove(path)

            if len(os.listdir(model_dir)) > 0:
                raise Exception("Couldn't delete all files.")
        elif len(os.listdir(model_dir)) > 0:
            raise Exception("There are files in this folder. Try setting delete_existing_files=True.")
    else:
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    if model_type == "gpt" or model_type == "starcoder":
        model_type = "gptnext"

    if model_type == "mixtral":
        model_type = "llama"

    tmp_dir = tempfile.TemporaryDirectory()
    nemo_export_dir = Path(tmp_dir.name)

    model, model_config, tokenizer = load_nemo_model(nemo_checkpoint_path, nemo_export_dir)
    weights_dicts, model_configs = model_to_trtllm_ckpt(
        model=model,
        nemo_model_config=model_config,
        nemo_export_dir=nemo_export_dir,
        decoder_type=model_type,
        dtype=dtype,
        tensor_parallel_size=tensor_parallelism_size,
        pipeline_parallel_size=pipeline_parallelism_size,
        gpus_per_node=gpus_per_node,
        use_parallel_embedding=use_parallel_embedding,
        use_embedding_sharing=use_embedding_sharing,
    )

    for weight_dict, model_config in zip(weights_dicts, model_configs):
        rank = model_config.mapping.tp_rank
        for k, v in weight_dict.items():
            if isinstance(v, np.ndarray):
                v = numpy_to_torch(v)
            weight_dict[k] = v

        safetensors.torch.save_file(weight_dict, os.path.join(model_dir, f'rank{rank}.safetensors'))
    model_configs[0].to_json_file(os.path.join(model_dir, 'config.json'))

    tokenizer_path = os.path.join(nemo_export_dir, "tokenizer.model")
    if os.path.exists(tokenizer_path):
        shutil.copy(tokenizer_path, model_dir)
    else:
        if tokenizer is not None:
            tokenizer.save_pretrained(model_dir)

    nemo_model_config = os.path.join(nemo_export_dir, "model_config.yaml")
    if os.path.exists(nemo_model_config):
        shutil.copy(nemo_model_config, model_dir)

    tmp_dir.cleanup()
