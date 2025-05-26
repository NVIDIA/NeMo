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


"""Utilities for adapter testing."""

import multiprocessing
import socket
import time

import pytest
from flask import Flask, jsonify, request

from nemo.collections.llm.evaluation.adapters.server import AdapterServer

DEFAULT_FAKE_RESPONSE = {
    "object": "chat.completion",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "This is a fake LLM response</think>This survives reasoning",
            }
        }
    ],
}


def is_port_open(host, port, timeout=0.5):
    """Check if the given port is open on the host."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0


def wait_for_server(host, port, max_wait=5, interval=0.2):
    """Wait for server to be ready with timeout."""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if is_port_open(host, port):
            return True
        time.sleep(interval)
    return False


def create_and_run_adapter_server(url, config) -> AdapterServer:
    """Create and run an AdapterServer in a separate process."""
    adapter = AdapterServer(api_url=url, adapter_config=config)
    adapter.run()
    return adapter


def create_and_run_fake_endpoint():
    """Create and run a fake OpenAI API endpoint."""
    app = Flask(__name__)

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completion():
        data = request.json
        if "fake_response" in data:
            response = data["fake_response"]
        else:
            response = DEFAULT_FAKE_RESPONSE
        return jsonify(response)

    app.run(host="localhost", port=3300)


def create_fake_endpoint_process():
    """Create a process running a fake OpenAI endpoint.

    Returns:
        The multiprocessing.Process object running the endpoint.
    """
    p = multiprocessing.Process(target=create_and_run_fake_endpoint)
    p.start()

    # Wait for the server to be ready
    if not wait_for_server("localhost", 3300):
        p.terminate()
        pytest.fail("Fake OpenAI endpoint did not start within the timeout period")

    return p


def create_adapter_server_process(api_url, adapter_config):
    """Create a process that runs an AdapterServer and return both the process and a local instance.

    Args:
        api_url: The API URL the adapter will call.
        output_dir: Directory for output files.
        adapter_config: The configuration for the adapter.

    Returns:
        A tuple of (process, adapter_instance) where adapter_instance is a local
        instance with the same configuration.
    """

    adapter = AdapterServer(api_url=api_url, adapter_config=adapter_config)
    p = multiprocessing.Process(target=adapter.run)
    p.start()

    # Wait for the server to be ready
    if not wait_for_server("localhost", adapter.adapter_port):
        p.terminate()
        pytest.fail(f"Server did not start within the timeout period on port {adapter.adapter_port}")

    return p, adapter
