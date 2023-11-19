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
import uvicorn
from nemo.utils import logging
from service.rest_model_api import app

def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Start REST service for OpenAI API type request",
    )

    parser.add_argument(
        "-sha",
        "--service_http_address",
        default="0.0.0.0",
        type=str,
        help="HTTP address for the REST Service"
    )

    parser.add_argument(
        "-sp",
        "--service_port",
        default=8000,
        type=int,
        help="Port for the Triton server to listen for requests"
    )

    parser.add_argument(
        "-tha",
        "--triton_http_address",
        default="0.0.0.0",
        type=str,
        help="HTTP address for the Triton server"
    )

    parser.add_argument(
        "-tp",
        "--triton_port",
        default=9500,
        type=int,
        help="Port for the Triton server to listen for requests"
    )

    parser.add_argument(
        "-tmv",
        "--triton_model_version",
        default=1,
        type=int,
        help="Version for the service"
    )

    parser.add_argument(
        "-dm",
        "--debug_mode",
        default="False",
        type=str,
        help="Enable debug mode"
    )

    args = parser.parse_args(argv)
    return args


def start_rest_service(argv):
    args = get_args(argv)

    if args.debug_mode == "True":
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    logging.setLevel(loglevel)
    logging.info("Logging level set to {}".format(loglevel))
    logging.info(args)

    uvicorn.run(
        'service.rest_model_api:app',
        host=args.service_http_address,
        port=args.service_port,
        reload=True
    )

if __name__ == '__main__':
    start_rest_service(sys.argv[1:])
