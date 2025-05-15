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

use_ray = True
try:
    from ray import serve
except Exception:
    use_ray = False

import asyncio
import logging
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from .megatronllm_deployable import MegatronLLMDeployableNemo2

LOGGER = logging.getLogger("NeMo")

app = FastAPI()


@serve.deployment(
    num_replicas=1,  # One replica per GPU
    ray_actor_options={
        "num_gpus": 1,  # Each replica gets 1 GPU
        "num_cpus": 8,
    },
    max_ongoing_requests=10,
)
@serve.ingress(app)
class MegatronLLMRayDeployable:
    """A Ray Serve compatible wrapper for deploying Megatron LLM models.

    This class provides a standardized interface for deploying Megatron LLM models
    in Ray Serve. It supports various NLP tasks and handles model loading,
    inference, and deployment configurations.

    Args:
        nemo_checkpoint_filepath (str): Path to the .nemo checkpoint file.
        num_devices (int): Number of devices to use for deployment.
        num_nodes (int): Number of nodes to use for deployment.
        tensor_model_parallel_size (int): Size of the tensor model parallelism.
        pipeline_model_parallel_size (int): Size of the pipeline model parallelism.
        context_parallel_size (int): Size of the context parallelism.
        model_id (str): Identifier for the model in the API responses. Defaults to "nemo-model".
    """

    def __init__(
        self,
        nemo_checkpoint_filepath: str,
        num_devices: int = 1,
        num_nodes: int = 1,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        model_id: str = "nemo-model",
    ):
        """Initialize the Megatron LLM model deployment.

        Args:
            nemo_checkpoint_filepath (str): Path to the .nemo checkpoint file.
            num_devices (int): Number of devices to use for deployment.
            num_nodes (int): Number of nodes to use for deployment.
            tensor_model_parallel_size (int): Size of the tensor model parallelism.
            pipeline_model_parallel_size (int): Size of the pipeline model parallelism.
            context_parallel_size (int): Size of the context parallelism.
            model_id (str): Model identifier. Defaults to "nemo-model".

        Raises:
            ImportError: If Ray is not installed.
            Exception: If model initialization fails.
        """
        if not use_ray:
            raise ImportError("Ray is not installed")
        try:
            self.model = MegatronLLMDeployableNemo2(
                nemo_checkpoint_filepath=nemo_checkpoint_filepath,
                num_devices=num_devices,
                num_nodes=num_nodes,
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=pipeline_model_parallel_size,
                context_parallel_size=context_parallel_size,
            )
            self.model_id = model_id
        except Exception as e:
            LOGGER.error(f"Error initializing MegatronLLMServe replica: {str(e)}")
            raise

    @app.post("/v1/completions/")
    async def completions(self, request: Dict[Any, Any]):
        """Handle text completion requests.

        This endpoint processes text completion requests in OpenAI API format and returns
        generated completions with token usage information.

        Args:
            request (Dict[Any, Any]): Request dictionary containing:
                - prompts: List of input prompts
                - max_tokens: Maximum tokens to generate (optional)
                - temperature: Sampling temperature (optional)
                - top_k: Top-k sampling parameter (optional)
                - top_p: Top-p sampling parameter (optional)
                - model: Model identifier (optional)

        Returns:
            Dict containing:
                - id: Unique completion ID
                - object: Response type ("text_completion")
                - created: Timestamp
                - model: Model identifier
                - choices: List of completion choices
                - usage: Token usage statistics

        Raises:
            HTTPException: If inference fails.
        """
        try:
            loop = asyncio.get_event_loop()
            model_name = request.get('model', 'nemo-model')
            if "prompt" in request:
                request["prompts"] = [request["prompt"]]
            # Prepare inference parameters
            inference_inputs = {
                "prompts": request.get("prompts", []),
                "max_length": request.get("max_tokens", 256),
                "temperature": request.get("temperature", 1.0),
                "top_k": request.get("top_k", 1),
                "top_p": 0.0,
                "compute_logprob": False,
                "apply_chat_template": False,
            }
            # Run model inference in the thread pool
            results = await loop.run_in_executor(None, self.model.ray_infer_fn, inference_inputs)

            # Extract generated texts from results
            generated_texts = results["sentences"]

            # Calculate token counts asynchronously
            prompt_tokens = sum(len(p.split()) for p in inference_inputs["prompts"])
            completion_tokens = sum(len(r.split()) for r in generated_texts)
            total_tokens = prompt_tokens + completion_tokens

            output = {
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "text": " ".join(generated_texts),
                        "index": 0,
                        "logprobs": results.get("log_probs"),  # Now we can include logprobs if they were requested
                        "finish_reason": (
                            "length" if len(generated_texts[0]) >= inference_inputs["max_length"] else "stop"
                        ),
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }
            return output
        except Exception as e:
            LOGGER.error(f"Error during inference: {str(e)}")
            LOGGER.error(f"Request: {request}")
            raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

    @app.post("/v1/chat/completions/")
    async def chat_completions(self, request: Dict[Any, Any]):
        """Handle chat completion requests.

        This endpoint processes chat completion requests in OpenAI API format and returns
        generated responses with token usage information.

        Args:
            request (Dict[Any, Any]): Request dictionary containing:
                - messages: List of chat messages
                - max_tokens: Maximum tokens to generate (optional)
                - temperature: Sampling temperature (optional)
                - top_k: Top-k sampling parameter (optional)
                - top_p: Top-p sampling parameter (optional)
                - model: Model identifier (optional)

        Returns:
            Dict containing:
                - id: Unique chat completion ID
                - object: Response type ("chat.completion")
                - created: Timestamp
                - model: Model identifier
                - choices: List of chat completion choices
                - usage: Token usage statistics

        Raises:
            HTTPException: If inference fails.
        """
        try:
            # Extract parameters from the request dictionary
            messages = request.get('messages', [])

            # Convert messages to a single prompt
            prompt = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages])
            prompt += "\nassistant:"

            # Prepare inference parameters
            inference_inputs = {
                "prompts": [prompt],
                "max_length": request.get("max_tokens", 256),
                "temperature": request.get("temperature", 1.0),
                "top_k": request.get("top_k", 1),
                "top_p": request.get("top_p", 0.0),
                "compute_logprob": False,
                "apply_chat_template": False,  # We already applied the template
            }

            loop = asyncio.get_event_loop()

            # Run model inference in the thread pool
            results = await loop.run_in_executor(None, self.model.ray_infer_fn, inference_inputs)

            # Extract generated texts from results
            generated_texts = results["sentences"]

            # Calculate token counts
            prompt_tokens = len(prompt.split())
            completion_tokens = sum(len(r.split()) for r in generated_texts)
            total_tokens = prompt_tokens + completion_tokens

            output = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.get('model', 'nemo-model'),
                "choices": [
                    {
                        "message": {"role": "assistant", "content": generated_texts},
                        "index": 0,
                        "finish_reason": (
                            "length" if len(generated_texts[0]) >= inference_inputs["max_length"] else "stop"
                        ),
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }
            return output
        except Exception as e:
            LOGGER.error(f"Error during chat completion: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during chat completion: {str(e)}")

    @app.get("/v1/models")
    async def list_models(self):
        """List available models.

        Returns:
            Dict containing model information.
        """
        return {
            "data": [
                {
                    "id": self.model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "nvidia",
                }
            ],
            "object": "list",
        }

    @app.get("/v1/health")
    async def health_check(self):
        """Health check endpoint.

        Returns:
            Dict containing health status.
        """
        return {"status": "healthy"}
