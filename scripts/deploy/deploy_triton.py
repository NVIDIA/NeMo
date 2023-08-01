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

from nemo.utils import logging
from nemo.deploy import DeployPyTriton, NemoQuery
from nemo.export import TensorRTLLM

try:
    from contextlib import nullcontext
except ImportError:
    # handle python < 3.7
    from contextlib import suppress as nullcontext


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=f"Export NeMo models to ONNX/Torchscript",
    )
    parser.add_argument("nemo_checkpoint", help="Source .nemo file")
    parser.add_argument("service_name", help="Name for the service")
    parser.add_argument("--dtype", default="bf16", help="dtype of the model on TRT-LLM")
    parser.add_argument("--optimized", action="store_true", help="Use TRT-LLM for inference")
    parser.add_argument("--verbose", default=None, help="Verbose level for logging, numeric")

    args = parser.parse_args(argv)
    return args


def nemo_deploy(argv):
    args = get_args(argv)
    loglevel = logging.INFO
    # assuming loglevel is bound to the string value obtained from the
    # command line argument. Convert to upper case to allow the user to
    # specify --log=DEBUG or --log=debug
    if args.verbose is not None:
        numeric_level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % numeric_level)
        loglevel = numeric_level
    logging.setLevel(loglevel)
    logging.info("Logging level set to {}".format(loglevel))

    nm = None
    if args.optimized:
        Path(model_info["trt_llm_model_dir"]).mkdir(parents=True, exist_ok=True)
        trt_llm_exporter = TensorRTLLM(model_dir=model_info["trt_llm_model_dir"])
        trt_llm_exporter.export(nemo_checkpoint_path=model_info["checkpoint"], n_gpus=1)
        nm = DeployPyTriton(model=trt_llm_exporter, triton_model_name=model_name)
    else:
        nm = DeployPyTriton(model=trt_llm_exporter, triton_model_name=model_name)
        nm = DeployPyTriton(checkpoint_path=args.nemo_checkpoint, triton_model_name=args.service_name)

    nm.deploy()

    try:
        logging.info("Triton deploy function is called.")
        nm.serve()
        logging.info("Model is being served.")
    except:
        logging.info("An error has occurred and will stop serving the model.")

    nm.stop()


if __name__ == '__main__':
    nemo_deploy(sys.argv[1:])
