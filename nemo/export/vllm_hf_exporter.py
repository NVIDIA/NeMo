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


import tempfile
from pathlib import Path
from typing import List, Union, Optional
from multiprocessing import Process

import numpy as np
from pytriton.decorators import batch, first_value
from pytriton.model_config import Tensor
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from nemo.deploy import ITritonDeployable
from nemo.deploy.utils import cast_output, str_ndarray2list
from nemo.export.huggingface import export_to_hf
from nemo.utils import logging

AnyPath = Union[Path, str]


class vLLMHFExporter(ITritonDeployable):
    """
    The Exporter class uses vLLM APIs to convert a NeMo or HF model to vLLM and makes the class,
    deployable with Triton server.

    Example:
        from nemo.export import vLLMHFExporter
        from nemo.deploy import DeployPyTriton

        exporter = vLLMHFExporter()
        exporter.export(model="/path/to/model/")

        server = DeployPyTriton(
            model=exporter,
            triton_model_name='model'
        )

        server.deploy()
        server.serve()
        server.stop()
    """

    def __init__(self):
        self.model = None
        self.lora_models = None

    def export(self,
                model: Optional[str] = None,
                model_dir: Optional[AnyPath] = None,
                enable_lora: bool = False,

                nemo_checkpoint: Optional[str] = None,
                model_type: Optional[str] = None,
                device: Optional[str] = None,
                tensor_parallel_size: int = 1,
                pipeline_parallel_size: int = 1,
                max_model_len: Optional[int] = None,
                lora_checkpoints: Optional[List[str]] = None,
                dtype: str = 'auto',
                seed: int = 0,
                log_stats: bool = True,
                weight_storage: str = 'auto',
                gpu_memory_utilization: float = 0.9,
                quantization: Optional[str] = None,
                delete_existing_files: bool = True, # TODO: change to False
                **kwargs):
        """
        Exports the HF checkpoint to vLLM and initializes the engine.
        Args:
            model (str): model name or the path
            model_dir (str | Path): the directory to save the HF model, used for NeMo export
            enable_lora (bool): whether to enable lora
            **kwargs: additional arguments for the LLM constructor
        """
        model_dir = None
        
        if nemo_checkpoint is not None:
            logging.warning(f"nemo_checkpoint parameter is deprecated, use model instead")
            if model is None:
                model = nemo_checkpoint
        if model is None:
            raise ValueError("model parameter is required")
        if model_type is not None:
            logging.warning(f"model_type parameter is deprecated")
        if lora_checkpoints is not None:
            raise ValueError("lora_checkpoints parameter is not supported yet, please use the vLLMExporter")
        if log_stats is not None:
            logging.warning(f"log_stats parameter is deprecated")
        if weight_storage is not None:
            logging.warning(f"weight_storage parameter is deprecated")
        if device is not None:
            logging.warning(f"device parameter is deprecated")


        model_path = Path(model)
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Convert the NeMo 2 model to HF format
            if (model_path / 'context').exists():
                # Model loading (torch.distributed.load) conflict with vLLM distributed environment
                # The export is run in a separate process to avoid the conflict
                p = Process(target=export_to_hf, args=(model_path, model_dir or tmp_dir))
                p.start()
                p.join()
                model = str(model_dir or tmp_dir)

            self.model = LLM(model=model,
                             quantization=quantization,
                             enable_lora=enable_lora,
                             gpu_memory_utilization=gpu_memory_utilization,
                             dtype=dtype,
                             seed=seed,
                             tensor_parallel_size=tensor_parallel_size,
                             pipeline_parallel_size=pipeline_parallel_size,
                             max_model_len=max_model_len,
                             **kwargs)

    def add_lora_models(self, lora_model_name, lora_model):
        """
        Add a lora model to the exporter.
        """
        if self.lora_models is None:
            self.lora_models = {}
        self.lora_models[lora_model_name] = lora_model

    @property
    def get_triton_input(self):
        inputs = (
            Tensor(name="prompts", shape=(-1,), dtype=bytes),
            Tensor(name="max_output_len", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_k", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_p", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="temperature", shape=(-1,), dtype=np.single, optional=True),
        )
        return inputs

    @property
    def get_triton_output(self):
        outputs = (Tensor(name="outputs", shape=(-1,), dtype=bytes),)
        return outputs

    @batch
    @first_value("max_output_len", "top_k", "top_p", "temperature")
    def triton_infer_fn(self, **inputs: np.ndarray):
        try:
            infer_input = {"input_texts": str_ndarray2list(inputs.pop("prompts"))}
            if "max_output_len" in inputs:
                infer_input["max_output_len"] = inputs.pop("max_output_len")
            if "top_k" in inputs:
                infer_input["top_k"] = inputs.pop("top_k")
            if "top_p" in inputs:
                infer_input["top_p"] = inputs.pop("top_p")
            if "temperature" in inputs:
                infer_input["temperature"] = inputs.pop("temperature")

            output_texts = self.forward(**infer_input)
            output = cast_output(output_texts, np.bytes_)
        except Exception as error:
            err_msg = "An error occurred: {0}".format(str(error))
            output = cast_output([err_msg], np.bytes_)

        return {"outputs": output}

    _first_pass = True
    def forward(
        self,
        input_texts: List[str],
        max_output_len: int = 64,
        top_k: int = 1,
        top_p: float = 0.1,
        temperature: float = 1.0,
        lora_model_name: str = None,
        
        lora_uids: Optional[List[str]] = None,
        stop_words_list: Optional[List[str]] = None,
        bad_words_list: Optional[List[str]] = None,
        no_repeat_ngram_size: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
        prompt_embeddings_table: Optional[List[str]] = None,
        prompt_embeddings_checkpoint_path: Optional[str] = None,
        output_log_probs: bool = False,
        output_generation_logits: bool = False,
        output_context_logits: bool = False,
        streaming: Optional[bool] = False,
    ):
        """
        Forward pass for the model.
        """

        # TODO: SUPPORT LORA
        if lora_uids is not None and lora_uids != []:
            raise NotImplementedError("lora_uids is not supported")


        if stop_words_list is not None and stop_words_list != []:
            raise NotImplementedError("stop_words_list is not supported")

        if bad_words_list is not None and bad_words_list != []:
            raise NotImplementedError("bad_words_list is not supported")

        if no_repeat_ngram_size is not None:
            raise NotImplementedError("no_repeat_ngram_size is not supported")

        if task_ids is not None and task_ids != []:
            raise NotImplementedError("task_ids is not supported")

        if prompt_embeddings_table is not None:
            raise NotImplementedError("prompt_embeddings_table is not supported")

        if prompt_embeddings_checkpoint_path is not None:
            raise NotImplementedError("prompt_embeddings_checkpoint_path is not supported")

        if output_log_probs:
            raise NotImplementedError("output_log_probs is not supported")

        if output_generation_logits:
            raise NotImplementedError("output_generation_logits is not supported")

        if output_context_logits:
            raise NotImplementedError("output_context_logits is not supported")

        if streaming is not None and self._first_pass:
            logging.warning("streaming is not supported")
        
        if top_p == 0.:
            if self._first_pass:
                logging.warning("top_p must be greater than 0, defaulting to 0.1")
            top_p = 0.1
            

        self._first_pass = False

        assert self.model is not None, "Model is not initialized."

        lora_request = None
        if lora_model_name is not None:
            if self.lora_models is None:
                raise Exception("No lora models are available.")
            assert lora_model_name in self.lora_models.keys(), "Lora model was not added before"
            lora_request = LoRARequest(lora_model_name, 1, self.lora_models[lora_model_name])

        sampling_params = SamplingParams(
            max_tokens=max_output_len, temperature=temperature, top_k=int(top_k), top_p=top_p
        )

        request_output = self.model.generate(input_texts, sampling_params, lora_request=lora_request)
        output = []
        for o in request_output:
            output.append(o.outputs[0].text)

        return output
