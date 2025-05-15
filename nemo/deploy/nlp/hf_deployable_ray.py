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

from .hf_deployable import HuggingFaceLLMDeploy

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
class HFRayDeployable:
    """A Ray Serve compatible wrapper for deploying HuggingFace models.

    This class provides a standardized interface for deploying HuggingFace models
    in Ray Serve. It supports various NLP tasks and handles model loading,
    inference, and deployment configurations.

    Args:
        hf_model_id_path (str): Path to the HuggingFace model or model identifier.
            Can be a local path or a model ID from HuggingFace Hub.
        task (str): HuggingFace task type (e.g., "text-generation"). Defaults to "text-generation".
        trust_remote_code (bool): Whether to trust remote code when loading the model. Defaults to True.
        device_map (str): Device mapping strategy for model placement. Defaults to "auto".
        tp_plan (str): Tensor parallelism plan for distributed inference. Defaults to None.
        model_id (str): Identifier for the model in the API responses. Defaults to "nemo-model".
    """

    def __init__(
        self,
        hf_model_id_path: str,
        task: str = "text-generation",
        trust_remote_code: bool = True,
        model_id: str = "nemo-model",
    ):
        """Initialize the HuggingFace model deployment.

        Args:
            hf_model_id_path (str): Path to the HuggingFace model or model identifier.
            task (str): HuggingFace task type. Defaults to "text-generation".
            trust_remote_code (bool): Whether to trust remote code. Defaults to True.
            device_map (str): Device mapping strategy. Defaults to "auto".
            tp_plan (str): Tensor parallelism plan. Defaults to None.
            model_id (str): Model identifier. Defaults to "nemo-model".

        Raises:
            ImportError: If Ray is not installed.
            Exception: If model initialization fails.
        """
        if not use_ray:
            raise ImportError("Ray is not installed")
        try:
            self.model = HuggingFaceLLMDeploy(
                hf_model_id_path=hf_model_id_path, task=task, trust_remote_code=trust_remote_code
            )
            self.model_id = model_id
        except Exception as e:
            LOGGER.error(f"Error initializing HuggingFaceLLMServe replica: {str(e)}")
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
            # Run tokenization and model inference in the thread pool
            results = await loop.run_in_executor(
                None, self.model.ray_infer_fn, request  # Use default ThreadPoolExecutor
            )
            # Extract generated texts from results
            generated_texts = results.get("sentences", [])

            # Calculate token counts asynchronously
            prompt_tokens = sum(len(p.split()) for p in request.get("prompts", []))
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
                        "logprobs": (
                            {
                                "token_logprobs": results.get("logits", None),
                                "top_logprobs": results.get("scores", None),
                            }
                            if results.get("logits") is not None
                            else None
                        ),
                        "finish_reason": (
                            "length" if len(generated_texts[0]) >= request.get('max_tokens', 50) else "stop"
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

            # Convert prompt to list and add to request
            request["prompts"] = [prompt]

            loop = asyncio.get_event_loop()

            # Run tokenization and model inference in the thread pool
            results = await loop.run_in_executor(
                None, self.model.ray_infer_fn, request  # Use default ThreadPoolExecutor
            )

            # Extract generated texts from results
            generated_texts = results.get("sentences", [])

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
                            "length" if len(generated_texts[0]) >= request.get('max_tokens', 50) else "stop"
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

        This endpoint returns information about the deployed model in OpenAI API format.

        Returns:
            Dict containing:
                - object: Response type ("list")
                - data: List of model information
        """
        return {"object": "list", "data": [{"id": self.model_id, "object": "model", "created": int(time.time())}]}

    @app.get("/v1/health")
    async def health_check(self):
        """Check the health status of the service.

        This endpoint is used to verify that the service is running and healthy.

        Returns:
            Dict containing:
                - status: Health status ("healthy")
        """
        return {"status": "healthy"}
