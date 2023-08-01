# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import os
import pprint
import shutil
from pathlib import Path
import typing

import numpy as np
import torch
from pytriton.decorators import batch
from pytriton.model_config import Tensor

from nemo.deploy import ITritonDeployable
from nemo.deploy.utils import str_ndarray2list, cast_output

from .trt_llm.nemo_utils import get_model_config, get_tokenzier, nemo_decode, nemo_to_tensorrt_llm
from .trt_llm.tensorrt_llm_run import generate, load


class TensorRTLLM(ITritonDeployable):
    def __init__(self, model_dir: str):
        if not Path(model_dir).is_dir():
            raise Exception("A valid directory path should be provided.")

        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self):
        self.tokenizer = None
        self.model = None

        folders = os.listdir(self.model_dir)
        if len(folders) > 0:
            for f in folders:
                if Path(os.path.join(self.model_dir, f)).is_dir():
                    if f[-3:] == "gpu":
                        self.tokenizer = get_tokenzier(os.path.join(self.model_dir, f))
                        self.model = load(tokenizer=self.tokenizer, engine_dir=self.model_dir)

    def export(
        self,
        nemo_checkpoint_path,
        delete_existing_files=True,
        n_gpus=1,
        max_input_len=200,
        max_output_len=200,
        max_batch_size=32,
    ):
        if delete_existing_files and len(os.listdir(self.model_dir)) > 0:
            for files in os.listdir(self.model_dir):
                path = os.path.join(self.model_dir, files)
                try:
                    shutil.rmtree(path)
                except OSError:
                    os.remove(path)

            if len(os.listdir(self.model_dir)) > 0:
                raise Exception("Couldn't delete all files.")
        elif len(os.listdir(self.model_dir)) > 0:
            raise Exception("There are files in this folder. Try setting delete_existing_files=True.")

        self.model = None

        weights_dir, model_config, tokenizer = nemo_decode(
            nemo_checkpoint_path, self.model_dir, tensor_parallelism=n_gpus
        )

        model_config = get_model_config(weights_dir)
        nemo_to_tensorrt_llm(
            weights_dir,
            model_config,
            self.model_dir,
            gpus=n_gpus,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_batch_size=max_batch_size,
        )

        self._load()

    def forward(self, input_texts, max_output_len=200):
        if self.model is None:
            raise Exception(
                "A nemo checkpoint should be exported and " "TensorRT LLM should be loaded first to run inference."
            )
        else:
            return generate(input_texts, max_output_len, self.model)

    @property
    def get_triton_input(self):
        inputs = (Tensor(name="prompts", shape=(1,), dtype=bytes),)
        return inputs

    @property
    def get_triton_output(self):
        outputs = (Tensor(name="outputs", shape=(1,), dtype=bytes),)
        return outputs

    @batch
    def triton_infer_fn(self, **inputs: np.ndarray):
        input_texts = str_ndarray2list(inputs.pop("prompts"))
        output_texts = self.forward(input_texts)
        output = cast_output(output_texts, np.bytes_)
        return {"outputs": output}
