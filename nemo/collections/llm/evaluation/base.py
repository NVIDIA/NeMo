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

import importlib
import pkgutil
import re

try:
    import core_evals
except ImportError:
    raise ImportError("Please ensure that core_evals is installed in your env as it is required to run evaluations")

from nemo.utils import logging


def wait_for_server_ready(
    url: str = 'http://0.0.0.0:8000',
    triton_http_port: int = 8000,
    model_name: str = 'triton_model',
    max_retries: int = 600,
    retry_interval: int = 2,
):
    """
    Wait for PyTriton server and model to be ready.

    Args:
        url (str): The URL of the Triton server (e.g., "grpc://0.0.0.0:8001").
        triton_http_port (int): http port of the triton server.
        model_name (str): The name of the deployed model.
        max_retries (int): Maximum number of retries before giving up.
        retry_interval (int): Time in seconds to wait between retries.

    Returns:
        bool: True if both the server and model are ready within the retries, False otherwise.
    """

    import time

    import requests
    from pytriton.client import ModelClient
    from pytriton.client.exceptions import PyTritonClientModelUnavailableError, PyTritonClientTimeoutError

    # If gRPC URL, extract HTTP URL from gRPC URL for health checks
    if url.startswith("grpc://"):
        # Extract the gRPC port using regex
        pattern = r":(\d+)"  # Matches a colon followed by one or more digits
        match = re.search(pattern, url)
        grpc_port = match.group(1)
        # Replace 'grpc' with 'http' and replace the grpc_port with http port
        url = url.replace("grpc://", "http://").replace(f":{grpc_port}", f":{triton_http_port}")
    health_url = f"{url}/v2/health/ready"

    for _ in range(max_retries):
        logging.info("Checking server and model readiness...")

        try:
            # Check server readiness using HTTP health endpoint
            response = requests.get(health_url)
            if response.status_code != 200:
                logging.info(f"Server is not ready. HTTP status code: {response.status_code}")
                time.sleep(retry_interval)
                continue
            logging.info("Server is ready.")

            # Check model readiness using ModelClient
            with ModelClient(url, model_name=model_name, init_timeout_s=retry_interval):
                logging.info(f"Model '{model_name}' is ready.")
                return True

        except PyTritonClientTimeoutError:
            logging.info(f"Timeout: Server or model '{model_name}' not ready yet.")
        except PyTritonClientModelUnavailableError:
            logging.info(f"Model '{model_name}' is unavailable on the server.")
        except requests.exceptions.RequestException:
            logging.info(f"Pytriton server not ready yet. Retrying in {retry_interval} seconds...")

        # Wait before retrying
        time.sleep(retry_interval)

    logging.error(f"Server or model '{model_name}' not ready after {max_retries} attempts.")
    return False


def _iter_namespace(ns_pkg):
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


def find_framework(eval_task: str) -> str:
    discovered_modules = {
        name: importlib.import_module('.input', package=name) for finder, name, ispkg in _iter_namespace(core_evals)
    }

    for framework_name, input_module in discovered_modules.items():
        _, task_name_mapping = input_module.get_available_tasks()
        if eval_task in task_name_mapping.keys():
            return framework_name

    raise ValueError(f"Framework for task {eval_task} not found!")
