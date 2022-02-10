#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

import os

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

import inference_lib.gpt_token_encoder as gpt_encoder


# Maximum tested sequence length
MAXIMUM_LENGTH = 200


def args_parser(parser):
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u", "--url", type=str, required=False, help="Inference server URL."
    )
    parser.add_argument(
        "-i",
        "--protocol",
        type=str,
        required=False,
        default="http",
        help=f'Protocol ({"|".join(PROTOCOL_MAP)})'
        + f'used to communicate with inference service. Default is "http".',
    )
    parser.add_argument(
        "-l",
        "--output-len",
        type=int,
        default=20,
        required=False,
        help="Specify output len",
    )
    parser.add_argument(
        "-d",
        "--datasets-dir",
        type=str,
        required=False,
        default="./",
        help="Folder contains vocab and dataset",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        required=False,
        default="fastertransformer",
        help="model name",
    )


def vocabulary_validate(encoder, url, model_name, protocol):
    config = query_configuration(url, model_name, protocol)
    end_id_triton = int(config["parameters"]["end_id"]["string_value"])
    end_id_encoder = encoder.end_id()
    if not end_id_triton == end_id_encoder:
        print("TRITON config END ID", end_id_triton)
        print("VOCABULARY END ID", end_id_encoder)
        print("ERROR: Triton END ID doesn't match vocabulary END ID")
        exit(1)
    return end_id_triton


def args_validate(parser):
    FLAGS = parser.parse_args()
    client_util, default_url = PROTOCOL_MAP.get(FLAGS.protocol, (None, None))
    if client_util is None:
        print(
            f'unexpected protocol "{FLAGS.protocol}',
            f'expects: {"|".join(PROTOCOL_MAP)}',
        )
        exit(1)

    if FLAGS.url is None:
        FLAGS.url = default_url

    if FLAGS.output_len > MAXIMUM_LENGTH:
        print(
            f"Output tokens length (--output-len, -l) value {FLAGS.output_len} not supported.",
            f"The maximum value is {MAXIMUM_LENGTH}.",
        )
        exit(1)

    if FLAGS.output_len < 1:
        print(
            f"Output tokens length (--output-len, -l) value {FLAGS.output_len} not supported.",
            f"The minimum value is 1.",
        )
        exit(1)

    # Prepare encoder and decoder
    merge_file = os.path.join(FLAGS.datasets_dir, "gpt2-merges.txt")
    vocab_file = os.path.join(FLAGS.datasets_dir, "gpt2-vocab.json")
    encoder = gpt_encoder.get_encoder(vocab_file, merge_file)

    end_id = vocabulary_validate(encoder, FLAGS.url, FLAGS.model_name, FLAGS.protocol)
    setattr(FLAGS, "end_id", end_id)

    return FLAGS, encoder, client_util


def query_configuration(infer_url, model_name, protocol):
    # curl <http://your.server:8000>/v2/models/<your.model>/config
    import requests
    import json

    infer_url_parts = infer_url.split(":")
    assert len(infer_url_parts) == 2
    node = infer_url_parts[0]

    prefix = "https" if protocol == "https" else "http"

    url = f"{prefix}://{node}:8000/v2/models/{model_name}/config"
    response = requests.get(url)
    if response.status_code != 200:
        print("ERROR: Model configuration can't be read from Triton")
        print(response)
        raise Exception("Network failure")
    else:
        return json.loads(response.content)


def send_requests(
    url,
    protocol,
    input_start_ids,
    input_len,
    output_len,
    verbose,
    client_util,
    model_name="fastertransformer",
):
    use_ssl = protocol == "https"
    with client_util.InferenceServerClient(url, verbose=verbose, ssl=use_ssl) as client:
        input_data = input_start_ids
        inputs = [
            client_util.InferInput(
                "INPUT_ID",  # Input tokens
                input_data.shape,
                np_to_triton_dtype(input_data.dtype),
            ),
            client_util.InferInput(
                "REQUEST_INPUT_LEN",
                input_len.shape,
                np_to_triton_dtype(input_len.dtype),
            ),
            client_util.InferInput(
                "REQUEST_OUTPUT_LEN",
                output_len.shape,
                np_to_triton_dtype(output_len.dtype),
            ),
        ]
        inputs[0].set_data_from_numpy(input_data)
        inputs[1].set_data_from_numpy(input_len)
        inputs[2].set_data_from_numpy(output_len)
        result = client.infer(model_name, inputs)

        return result.as_numpy("OUTPUT0")


PROTOCOL_MAP = {
    "http": (httpclient, "localhost:8000"),
    "https": (httpclient, "localhost:8000"),
    "grpc": (grpcclient, "localhost:8001"),
}
