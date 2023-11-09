# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from pytriton.client import ModelClient
import numpy as np


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Exports nemo models stored in nemo checkpoints to TensorRT-LLM",
    )

    parser.add_argument(
        "-u",
        "--url",
        required=True,
        type=str,
        help="url for the triton server"
    )

    parser.add_argument(
        "-mn",
        "--model_name",
        required=True,
        type=str,
        help="Name of the triton model"
    )

    parser.add_argument(
        "-p",
        "--prompt",
        required=True,
        type=str,
        help="Prompt"
    )

    parser.add_argument(
        "-mot",
        "--max_output_token",
        default=128,
        type=int,
        help="Max output token length"
    )

    parser.add_argument(
        "-tk",
        "--top_k",
        default=1,
        type=int,
        help="top_k"
    )

    parser.add_argument(
        "-tp",
        "--top_p",
        default=0.0,
        type=float,
        help="top_p"
    )

    parser.add_argument(
        "-t",
        "--temperature",
        default=1.0,
        type=float,
        help="temperature"
    )

    parser.add_argument(
        "-it",
        "--init_timeout",
        default=600.0,
        type=float,
        help="init timeout for the triton server"
    )

    args = parser.parse_args(argv)
    return args


def query_llm(url, model_name, prompts, max_output_token=128, top_k=1, top_p=0.0, temperature=1.0,  init_timeout=600.0):
    str_ndarray = np.array(prompts)[..., np.newaxis]
    prompts = np.char.encode(str_ndarray, "utf-8")
    max_output_token = np.full(prompts.shape, max_output_token, dtype=np.int_)
    top_k = np.full(prompts.shape, top_k, dtype=np.int_)
    top_p = np.full(prompts.shape, top_p, dtype=np.single)
    temperature = np.full(prompts.shape, temperature, dtype=np.single)

    with ModelClient(url, model_name, init_timeout_s=init_timeout) as client:
        result_dict = client.infer_batch(
            prompts=prompts,
            max_output_token=max_output_token,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        output_type = client.model_config.outputs[0].dtype

    if output_type == np.bytes_:
        sentences = np.char.decode(result_dict["outputs"].astype("bytes"), "utf-8")
        return sentences
    else:
        return result_dict["outputs"]


def query(argv):
    args = get_args(argv)

    output = query_llm(
        url=args.url,
        model_name=args.model_name,
        prompts=[args.prompt],
        max_output_token=args.max_output_token,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        init_timeout=args.init_timeout,
    )

    print("output: ", output)

if __name__ == '__main__':
    query(sys.argv[1:])
