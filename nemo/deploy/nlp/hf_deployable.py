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
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
import wrapt
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline

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

SUPPORTED_TASKS = ["text-generation"]


class HuggingFaceLLMDeploy(ITritonDeployable):
    """A Triton inference server compatible wrapper for HuggingFace models.

    This class provides a standardized interface for deploying HuggingFace models
    in Triton inference server. It supports various NLP tasks and handles model
    loading, inference, and deployment configurations.

    Args:
        hf_model_id_path: Path to the HuggingFace model or model identifier.
            Can be a local path or a model ID from HuggingFace Hub.
        model: Pre-loaded HuggingFace model. If provided, hf_model_id_path will be ignored.
        tokenizer_id_path: Path to the tokenizer or tokenizer identifier.
            If None, will use the same path as hf_model_id_path.
        task: HuggingFace task type (e.g., "text-generation", "question-answering").
            Required if hf_model_id_path is provided.
        trust_remote_code: Whether to trust remote code when loading models.
            Should be True for custom models with custom code.
        device_id: GPU device ID to use. If None, uses current CUDA device.

    Raises:
        ValueError: If neither hf_model_id_path nor model is provided.
        ValueError: If hf_model_id_path is provided but task is not specified.
    """

    def __init__(
        self,
        hf_model_id_path: Optional[str] = None,
        tokenizer_id_path: Optional[str] = None,
        model: Optional[AutoModel] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        task: Optional[str] = "text-generation",
        trust_remote_code: bool = False,
        device_id: Optional[int] = None,
    ):
        if hf_model_id_path is None and model is None:
            raise ValueError("hf_model_id_path or model parameters has to be passed.")
        elif hf_model_id_path is not None and model is not None:
            LOGGER.warning(
                "hf_model_id_path will be ignored and the HuggingFace model " "set with model parameter will be used."
            )

        assert task in SUPPORTED_TASKS, "Task {0} is not a support task.".format(task)

        self.hf_model_id_path = hf_model_id_path
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_id_path = tokenizer_id_path
        self.trust_remote_code = trust_remote_code
        self.device_id = torch.cuda.current_device() if device_id is None else device_id

        if model is None:
            self._load()

    def _load(self) -> None:
        """Load the HuggingFace pipeline with the specified model and task.

        This method initializes the HuggingFace pipeline using the provided model
        configuration and task type. It handles the model and tokenizer loading
        process.

        Raises:
            AssertionError: If task is not specified.
        """
        assert self.task is not None, "A task has to be given for the generation task."

        if self.task == "text-generation":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_id_path,
                torch_dtype='auto',
                device_map="cpu",
                trust_remote_code=self.trust_remote_code,
            )
        else:
            raise ValueError("Task {0} is not supported.".format(self.task))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_id_path,
            trust_remote_code=self.trust_remote_code,
        )

    def generate(
        self,
        **kwargs: Any,
    ) -> List[str]:
        """Generate text based on the provided input prompts.

        This method processes input prompts through the loaded pipeline and
        generates text according to the specified parameters.

        Args:
            **kwargs: Generation parameters including:
                - text_inputs: List of input prompts
                - max_length: Maximum number of tokens to generate
                - num_return_sequences: Number of sequences to generate per prompt
                - temperature: Sampling temperature
                - top_k: Number of highest probability tokens to consider
                - top_p: Cumulative probability threshold for token sampling
                - do_sample: Whether to use sampling
                - return_full_text: Whether to return full text or only generated part

        Returns:
            List[str]: A list of generated texts, one for each input prompt.

        Raises:
            RuntimeError: If the pipeline is not initialized.
        """
        if not self.model:
            raise RuntimeError("Model is not initialized")

        inputs = self.tokenizer(kwargs["text_inputs"], return_tensors="pt")
        kwargs = {**inputs, **kwargs}
        kwargs.pop("text_inputs")
        generated_ids = self.model.generate(**kwargs)
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return output

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
            Tensor(name="output_logits", shape=(-1,), dtype=np.bool_, optional=True),
            Tensor(name="output_scores", shape=(-1,), dtype=np.bool_, optional=True),
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
            temperature = inputs.pop("temperature")[0][0] if "temperature" in inputs else 1.0
            top_k = int(inputs.pop("top_k")[0][0] if "top_k" in inputs else 1)
            top_p = inputs.pop("top_p")[0][0] if "top_k" in inputs else 0.0
            num_tokens_to_generate = inputs.pop("max_length")[0][0] if "max_length" in inputs else 256
            output_logits = inputs.pop("output_logits")[0][0] if "output_logits" in inputs else False
            output_scores = inputs.pop("output_scores")[0][0] if "output_scores" in inputs else False

            return_dict_in_generate = False
            if output_logits or output_scores:
                return_dict_in_generate = True

            output = self.generate(
                text_inputs=prompts,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=num_tokens_to_generate,
                output_logits=output_logits,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
            )

            output_infer = {"sentences": cast_output(output, np.bytes_)}

        except Exception as error:
            err_msg = "An error occurred: {0}".format(str(error))
            output_infer["sentences"] = cast_output([err_msg], np.bytes_)

        return output_infer
