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

import argparse
import json
import logging

import requests

LOGGER = logging.getLogger("NeMo")


def parse_args():
    parser = argparse.ArgumentParser(description="Query a deployed HuggingFace model using Ray")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address of the Ray Serve server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=1024,
        help="Port number of the Ray Serve server",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="nemo-model",
        help="Identifier for the model in the API responses",
    )
    return parser.parse_args()


def test_completions_endpoint(base_url: str, model_id: str) -> None:
    """Test the completions endpoint."""
    url = f"{base_url}/v1/completions/"
    payload = {
        "model": model_id,
        "prompts": ["Hello, how are you?"],
        "max_tokens": 50,
        "temperature": 0.7,
    }

    LOGGER.info(f"Testing completions endpoint at {url}")
    response = requests.post(url, json=payload)
    LOGGER.info(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        LOGGER.info(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        LOGGER.error(f"Error: {response.text}")


def test_chat_completions_endpoint(base_url: str, model_id: str) -> None:
    """Test the chat completions endpoint."""
    url = f"{base_url}/v1/chat/completions/"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 50,
        "temperature": 0.7,
    }

    LOGGER.info(f"Testing chat completions endpoint at {url}")
    response = requests.post(url, json=payload)
    LOGGER.info(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        LOGGER.info(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        LOGGER.error(f"Error: {response.text}")


def test_models_endpoint(base_url: str) -> None:
    """Test the models endpoint."""
    url = f"{base_url}/v1/models"

    LOGGER.info(f"Testing models endpoint at {url}")
    response = requests.get(url)
    LOGGER.info(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        LOGGER.info(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        LOGGER.error(f"Error: {response.text}")


def test_health_endpoint(base_url: str) -> None:
    """Test the health endpoint."""
    url = f"{base_url}/v1/health"

    LOGGER.info(f"Testing health endpoint at {url}")
    response = requests.get(url)
    LOGGER.info(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        LOGGER.info(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        LOGGER.error(f"Error: {response.text}")


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"

    LOGGER.info(f"Testing endpoints for model {args.model_id} at {base_url}")

    # Test all endpoints
    # test_health_endpoint(base_url)
    # test_models_endpoint(base_url)
    test_completions_endpoint(base_url, args.model_id)
    test_chat_completions_endpoint(base_url, args.model_id)


if __name__ == "__main__":
    main()
