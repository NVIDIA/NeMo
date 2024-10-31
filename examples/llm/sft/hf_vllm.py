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


from nemo.deploy import DeployPyTriton
from nemo.deploy.nlp import NemoQueryLLM
from nemo.export.vllm_hf_exporter import vLLMHFExporter


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument("-tmn", "--triton_model_name", required=True, type=str, help="Name for the service")
    parser.add_argument("-tmv", "--triton_model_version", default=1, type=int, help="Version for the service")
    parser.add_argument(
        "-trp", "--triton_port", default=8000, type=int, help="Port for the Triton server to listen for requests"
    )
    parser.add_argument(
        "-tha", "--triton_http_address", default="0.0.0.0", type=str, help="HTTP address for the Triton server"
    )
    parser.add_argument(
        "-trt", "--triton_request_timeout", default=60, type=int, help="Timeout in seconds for Triton server"
    )
    args = parser.parse_args()

    exporter = vLLMHFExporter()
    exporter.export(model=args.model)

    nm = DeployPyTriton(
        model=exporter,
        triton_model_name=args.triton_model_name,
        triton_model_version=args.triton_model_version,
        #max_batch_size=args.max_batch_size,
        port=args.triton_port,
        address=args.triton_http_address,
    )

    nm.deploy()
    nm.run()

    nq = NemoQueryLLM(url="localhost:8000", model_name=args.triton_model_name)

    output_deployed = nq.query_llm(
        prompts=["How are you doing?"],
        # max_output_len=max_output_len,
        top_k=1,
        top_p=0.2,
        temperature=1.0,
    )

    print("------------- Output: ", output_deployed)

    nm.stop()

    # subprocess.run(["deactivate"])
