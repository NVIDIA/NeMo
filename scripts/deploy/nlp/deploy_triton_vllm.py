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

import logging
logging.basicConfig()
LOGGER = logging.getLogger("NeMo")
LOGGER.setLevel(logging.INFO)

import argparse
import tempfile
import os.path

from nemo.export.vllm import Exporter
from nemo.deploy import DeployPyTriton


def main(args: argparse.Namespace):
    """
    Main function that exports a checkpoint into a vLLM-compatible format
    and deploys the model on the Triton server.
    """
    
    tempdir = None
    if args.model_dir is None:
        tempdir = tempfile.TemporaryDirectory()
        args.model_dir = tempdir.name
    elif not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    try:
        exporter = Exporter()
        exporter.export(
            nemo_checkpoint=args.nemo_checkpoint,
            model_dir=args.model_dir,
            model_type=args.model_type,
            tensor_parallel_size=args.num_gpus,
            weight_storage=args.weight_storage)
        
        server = DeployPyTriton(
            model=exporter,
            triton_model_name=args.triton_model_name,
            streaming=args.enable_streaming)
        
        server.deploy()
        server.serve()
        server.stop()
    finally:
        if tempdir is not None:
            tempdir.cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to export a NeMo checkpoint to vLLM and deploy it on Triton server.')
    parser.add_argument('-nc', '--nemo_checkpoint', required=True,
                        help='Path to the .nemo file with the model checkpoint')
    parser.add_argument('-md', '--model_dir', required=False,
                        help='Path to an intermediate directory where the converted files are stored. '
                        'The intermediate files can be cached between export runs.')
    parser.add_argument('-mt', '--model_type', required=True,
                        help='Model architecture, such as llama, mistral, mixtral.')
    parser.add_argument('-tmn', '--triton_model_name', required=True,
                        help='Name for the model deployed on the Triton server, arbitrary.')
    parser.add_argument('-ws', '--weight_storage', default='auto', choices=['auto', 'cache', 'file', 'memory'],
                        help='Strategy for storing converted weights: "file" - always write weights into a file, '
                        '"memory" - always do an in-memory conversion, "cache" - reuse existing files if they are '
                        'newer than the nemo checkpoint, "auto" - use "cache" for multi-GPU runs and "memory" '
                        'for single-GPU runs.')
    parser.add_argument('-es', '--enable_streaming', default=False, action='store_true',
                        help='Enable streaming responses through gRPC.')
    parser.add_argument('-ng', '--num_gpus', type=int, default=1,
                        help='Number of GPUs to use in tensor-parallel mode.')
    args = parser.parse_args()
    main(args)
