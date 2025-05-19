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


if __name__ == '__main__':
    args = get_args()

    if args.prompt_file is not None:
        with open(args.prompt_file, "r") as f:
            args.prompt = f.read()

    nemo_query = NemoQueryTRTLLMPytorch(args.url, args.model_name)

    outputs = nemo_query.query_llm(
        prompts=[args.prompt],
        max_length=args.max_length,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        init_timeout=args.init_timeout,
    )
    print(outputs)
