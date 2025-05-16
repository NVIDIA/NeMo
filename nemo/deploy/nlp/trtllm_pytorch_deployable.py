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
from typing import List, Optional, Union

import numpy as np
import torch
from pytriton.decorators import batch
from pytriton.model_config import Tensor
from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.llm import LLM, TokenizerBase
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from transformers import PreTrainedTokenizerBase

from nemo.deploy import ITritonDeployable
from nemo.deploy.utils import broadcast_list, cast_output, str_ndarray2list

LOGGER = logging.getLogger("NeMo")


class TensorRTLLMPyotrchDeployable(ITritonDeployable):
    """A Triton inference server compatible wrapper for TensorRT-LLM PyTorch backend.

    This class provides a standardized interface for deploying TensorRT-LLM PyTorch backend
    in Triton inference server. It handles model loading, inference, and deployment configurations.

    Args:
        hf_model_id_path (str): Path to the HuggingFace model or model identifier.
            Can be a local path or a model ID from HuggingFace Hub.
        tokenizer (Optional[Union[str, Path, TokenizerBase, PreTrainedTokenizerBase]]):
            Path to the tokenizer or tokenizer instance.
        tensor_parallel_size (int): Tensor parallelism size. Defaults to 1.
        pipeline_parallel_size (int): Pipeline parallelism size. Defaults to 1.
        moe_expert_parallel_size (int): MOE expert parallelism size. Defaults to -1.
        moe_tensor_parallel_size (int): MOE tensor parallelism size. Defaults to -1.
        max_batch_size (int): Maximum batch size. Defaults to 8.
        max_num_tokens (int): Maximum total tokens across all sequences in a batch. Defaults to 8192.
        dtype (str): Model data type. Defaults to "auto".
        **kwargs: Additional keyword arguments to pass to model loading.
    """

    def __init__(
        self,
        hf_model_id_path: str,
        tokenizer: Optional[Union[str, Path, TokenizerBase, PreTrainedTokenizerBase]] = None,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        moe_expert_parallel_size: int = -1,
        moe_tensor_parallel_size: int = -1,
        max_batch_size: int = 8,
        max_num_tokens: int = 8192,
        dtype: str = "auto",
        **kwargs,
    ):
        config_args = {}
        for k in list(kwargs.keys()):
            if k in PyTorchConfig.__annotations__.keys():
                config_args[k] = kwargs.pop(k)
        pytorch_config = PyTorchConfig(**config_args)

        self.model = LLM(
            model=hf_model_id_path,
            tokenizer=hf_model_id_path if tokenizer is None else tokenizer,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            moe_expert_parallel_size=moe_expert_parallel_size,
            moe_tensor_parallel_size=moe_tensor_parallel_size,
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            dtype=dtype,
            pytorch_backend_config=pytorch_config,
            **kwargs,
        )

    def generate(
        self,
        prompts: List[str],
        max_length: int = 256,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> List[str]:
        """Generate text based on the provided input prompts.

        This method processes input prompts through the loaded model and
        generates text according to the specified parameters.

        Args:
            prompts: List of input prompts
            max_length: Maximum number of tokens to generate. Defaults to 256.
            temperature: Sampling temperature. Defaults to None.
            top_k: Number of highest probability tokens to consider. Defaults to None.
            top_p: Cumulative probability threshold for token sampling. Defaults to None.
            **kwargs: Additional keyword arguments to sampling params.

        Returns:
            List[str]: A list of generated texts, one for each input prompt.

        Raises:
            RuntimeError: If the model is not initialized.
        """
        if not self.model:
            raise RuntimeError("Model is not initialized")

        sampling_params = SamplingParams(
            max_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs,
        )

        outputs = self.model.generate(
            inputs=prompts,
            sampling_params=sampling_params,
        )

        return [output.outputs[0].text for output in outputs]

    def generate_other_ranks(self):
        """
        Generate function for ranks other than the rank 0.
        """
        while True:
            message = torch.empty(1, dtype=torch.long, device="cuda")
            torch.distributed.broadcast(message, src=0)
            if message == 0:
                prompts = broadcast_list(data=[None], src=0)
                temperature, top_k, top_p, max_length = broadcast_list(data=[None], src=0)

                self.generate(
                    prompts=prompts,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_length=max_length,
                )
            else:
                return

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
            temperature = inputs.pop("temperature")[0][0] if "temperature" in inputs else None
            top_k = int(inputs.pop("top_k")[0][0]) if "top_k" in inputs else None
            top_p = inputs.pop("top_p")[0][0] if "top_p" in inputs else None
            max_length = inputs.pop("max_length")[0][0] if "max_length" in inputs else 256

            if torch.distributed.is_initialized():
                if torch.distributed.get_world_size() > 1:
                    torch.distributed.broadcast(torch.tensor([0], dtype=torch.long, device="cuda"), src=0)
                    broadcast_list(prompts, src=0)
                    broadcast_list(
                        data=[
                            temperature,
                            top_k,
                            top_p,
                            max_length,
                        ],
                        src=0,
                    )

            output = self.generate(
                prompts=prompts,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_length=max_length,
            )

            output_infer = {"sentences": cast_output(output, np.bytes_)}

        except Exception as error:
            err_msg = "An error occurred: {0}".format(str(error))
            output_infer["sentences"] = cast_output([err_msg], np.bytes_)

        return output_infer
