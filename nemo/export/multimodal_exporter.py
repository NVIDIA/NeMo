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

import logging
import os
import shutil
from pathlib import Path

from nemo.deploy import ITritonDeployable
from nemo.export.multimodal.build import build_trtllm_engine, build_visual_engine
from nemo.export.multimodal.run import MultimodalModelRunner

LOGGER = logging.getLogger("NeMo")


class MultimodalExporter(ITritonDeployable):
    """
    Exports nemo checkpoints to TensorRT and run fast inference.

    Example:
        from nemo.export import MultimodalExporter

        exporter = MultimodalExporter(model_dir="/path/for/model/files")
        exporter.export(
            visual_checkpoint_path="/path/for/nemo/checkpoint",
            model_type="neva",
            tensor_parallel_size=1,
        )

        output = exporter.forward("Hi! What is in this image?", "/path/for/input_media")
        print("output: ", output)

    """

    def __init__(
        self,
        model_dir: str,
        load_model: bool = True,
    ):
        self.model_dir = model_dir
        self.runner = None

        if load_model:
            self._load()

    def export(
        self,
        visual_checkpoint_path: str,
        llm_checkpoint_path: str = None,
        model_type: str = "neva",
        llm_model_type: str = "llama",
        tensor_parallel_size: int = 1,
        max_input_len: int = 256,
        max_output_len: int = 256,
        max_batch_size: int = 1,
        max_multimodal_len: int = 1024,
        dtype: str = "bfloat16",
        delete_existing_files: bool = True,
        load_model: bool = True,
    ):
        if Path(self.model_dir).exists():
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
        else:
            Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        llm_dir = os.path.join(self.model_dir, "llm_engine")
        build_trtllm_engine(
            model_dir=llm_dir,
            visual_checkpoint_path=visual_checkpoint_path,
            llm_checkpoint_path=llm_checkpoint_path,
            model_type=model_type,
            llm_model_type=llm_model_type,
            tensor_parallel_size=tensor_parallel_size,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_batch_size=max_batch_size,
            max_multimodal_len=max_multimodal_len,
            dtype=dtype,
        )

        visual_dir = os.path.join(self.model_dir, "visual_engine")
        build_visual_engine(visual_dir, visual_checkpoint_path, model_type, max_batch_size)

        if load_model:
            self._load()

    def forward(
        self,
        input_text: str,
        input_media: str,
        max_output_len: int = 30,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        num_beams: int = 1,
    ):
        if self.runner is None:
            raise Exception(
                "A nemo checkpoint should be exported and " "then it should be loaded first to run inference."
            )

        input_media = self.runner.load_test_media(input_media)
        return self.runner.run(
            input_text, input_media, max_output_len, top_k, top_p, temperature, repetition_penalty, num_beams
        )

    def _load(self):
        llm_dir = os.path.join(self.model_dir, "llm_engine")
        visual_dir = os.path.join(self.model_dir, "visual_engine")
        self.runner = MultimodalModelRunner(visual_dir, llm_dir)
