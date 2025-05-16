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
import sys

from nemo.deploy.nlp import NemoQueryTRTLLMPytorch


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Queries Triton server running a TensorRT-LLM PyTorch backend model",
    )
    parser.add_argument("-u", "--url", default="0.0.0.0", type=str, help="url for the triton server")
    parser.add_argument("-mn", "--model_name", required=True, type=str, help="Name of the triton model")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("-p", "--prompt", required=False, type=str, help="Prompt")
    prompt_group.add_argument("-pf", "--prompt_file", required=False, type=str, help="File to read the prompt from")
    parser.add_argument("-ml", "--max_length", default=256, type=int, help="Max output token length")
    parser.add_argument("-tk", "--top_k", type=int, help="top_k")
    parser.add_argument("-tpp", "--top_p", type=float, help="top_p")
    parser.add_argument("-t", "--temperature", type=float, help="temperature")
    parser.add_argument("-it", "--init_timeout", default=60.0, type=float, help="init timeout for the triton server")

    args = parser.parse_args()
    return args


def query_llm(
    url,
    model_name,
    prompts,
    max_length=256,
    top_k=None,
    top_p=None,
    temperature=None,
    init_timeout=60.0,
):
    """Query a TensorRT-LLM PyTorch backend model deployed on Triton Inference Server.

    Args:
        url (str): URL of the Triton Inference Server (e.g. localhost or IP address)
        model_name (str): Name of the model as deployed on Triton server
        prompts (List[str]): List of text prompts to send to the model
        max_length (int, optional): Maximum number of tokens to generate in the response. Defaults to 256.
        top_k (int, optional): Number of highest probability tokens to consider for sampling. Defaults to None.
        top_p (float, optional): Cumulative probability threshold for token sampling. Defaults to None.
        temperature (float, optional): Temperature for controlling randomness in sampling (higher = more random). Defaults to None.
        init_timeout (float, optional): Timeout in seconds when initializing connection to Triton server. Defaults to 60.0.

    Returns:
        List[str]: Generated text responses for each input prompt
    """
    nemo_query = NemoQueryTRTLLMPytorch(url, model_name)
    return nemo_query.query_llm(
        prompts=prompts,
        max_length=max_length,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        init_timeout=init_timeout,
    )


def query():
    args = get_args()

    if args.prompt_file is not None:
        with open(args.prompt_file, "r") as f:
            args.prompt = f.read()

    outputs = query_llm(
        url=args.url,
        model_name=args.model_name,
        prompts=[args.prompt],
        max_length=args.max_length,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        init_timeout=args.init_timeout,
    )
    print(outputs)


if __name__ == '__main__':
    query()
