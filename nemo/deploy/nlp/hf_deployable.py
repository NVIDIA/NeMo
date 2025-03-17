# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
import wrapt

from transformers import AutoModel, AutoTokenizer, pipeline

from nemo.deploy import ITritonDeployable
from nemo.deploy.utils import cast_output, str_ndarray2list


@wrapt.decorator
def noop_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


use_pytriton = True
batch = noop_decorator
try:
    from pytriton.decorators import batch
    from pytriton.model_config import Tensor
except Exception:
    use_pytriton = False

LOGGER = logging.getLogger("NeMo")


class HuggingFaceLLMDeploy(ITritonDeployable):
    """
    Triton inference server compatible deploy class for a HF model file

    Args:
        hf_model_id_path (str): path for the nemo checkpoint.
        model (AutoModel): number of GPUs.
        tokenizer : number of nodes.
        task (str): Hugging Face tasks such as question-answering, text-classification, etc.
        trust_remote_code (int): tensor parallelism.
    """

    def __init__(
        self,
        hf_model_id_path: str = None,
        model=None,
        tokenizer_id_path=None,
        task: str = None,
        trust_remote_code: bool = False,
        device_id=None,
    ):
        if hf_model_id_path is None and model is None:
            raise ValueError("hf_model_id_path or model parameters has to be passed.")
        elif hf_model_id_path is not None and model is not None:
            LOGGER.warning(
                "hf_model_id_path will be ignored and the HuggingFace model " "set with model parameter will be used."
            )

        self.hf_model_id_path = hf_model_id_path
        self.task = task
        self.model = model
        self.tokenizer_id_path = tokenizer_id_path
        self.trust_remote_code = trust_remote_code
        self.pipeline = None
        self.device_id = torch.cuda.current_device() if device_id is None else device_id

        if model is None:
            self._load()

    def _load(self):
        assert self.task is not None, "A task has to be given for the generation task."
        self.pipeline = pipeline(
            self.task,
            model=self.hf_model_id_path,
            tokenizer=self.tokenizer_id_path,
            device=self.device_id,
        )

    def generate(
        self,
        **kwargs,
    ):
        """
        Generates text based on the provided input prompts.

        Args:
            prompts (List[str]): A list of input strings.
            output_scores (bool): Whether to return output scores or not.

        Returns:
            Dict: A list containing the generated results.
        """

        output = self.pipeline(**kwargs)
        generated_text = []
        for o in output:
            generated_text.append(o[0]["generated_text"])

        return generated_text

    @property
    def get_triton_input(self):
        inputs = (
            Tensor(name="prompts", shape=(-1,), dtype=bytes),
            Tensor(name="max_length", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="max_batch_size", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_k", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_p", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="temperature", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="random_seed", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="max_length", shape=(-1,), dtype=np.int_, optional=True),
        )
        return inputs

    @property
    def get_triton_output(self):
        return (Tensor(name="sentences", shape=(-1,), dtype=bytes),)

    @batch
    def triton_infer_fn(self, **inputs: np.ndarray):
        output_infer = {}
        try:
            prompts = str_ndarray2list(inputs.pop("prompts"))
            max_batch_size = inputs.pop("max_batch_size")[0][0] if "max_batch_size" in inputs else 32
            random_seed = inputs.pop("random_seed")[0][0] if "random_seed" in inputs else None
            temperature = inputs.pop("temperature")[0][0] if "temperature" in inputs else 1.0
            top_k = int(inputs.pop("top_k")[0][0] if "top_k" in inputs else 1)
            top_p = inputs.pop("top_p")[0][0] if "top_k" in inputs else 0.0
            num_tokens_to_generate = inputs.pop("max_length")[0][0] if "max_length" in inputs else 256
            log_probs = inputs.pop("compute_logprob")[0][0] if "compute_logprob" in inputs else False
            text_only = True

            output = self.generate(
                text_inputs=prompts,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=num_tokens_to_generate,
                return_full_text=False,
            )

            output_infer = {"sentences": cast_output(output, np.bytes_)}

        except Exception as error:
            err_msg = "An error occurred: {0}".format(str(error))
            output_infer["sentences"] = cast_output([err_msg], np.bytes_)

        return output_infer
