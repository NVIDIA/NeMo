# -*- coding: utf-8 -*-
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


""" Script responsible for processing of module inputs and generating outputs.

Args:
    Format of the input JSON file (passed as --input_filename):
    {
        "name": "my_lenet",
        "id": "nemo.collections.cv.modules.trainables.feed_forward_network.FeedForwardNetwork",
        "module_type": "trainable",
        "arguments": [
            {
                "name": "input_size",
                "value": 10
            },
            {
                "name": "output_size",
                "value": 2
            }
        ],
        "inputs": [
            {
                "name": "inputs",
                "value": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "type": "Tensor"
            }
        ]
    }

Required fields: "id", "name", "arguments", "inputs"

Returns:
    Format of the output JSON file (indicated  as --output_filename):
    [
        {"name": "outputs", "value": [1.3470189571380615, 0.06865233182907104], "type": "Tensor"}
    ]
"""

import argparse
import importlib
import json

import torch
from scripts.integration.get_module_ports import instantiate_module

import nemo
from nemo.utils import logging


def get_module_outputs():
    """ Main function processing of module inputs and generating outputs """
    # Parse filename.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input_filename',
        '-i',
        help='Name of the input JSON file containing module description and arguments',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--output_filename',
        '-o',
        help='Name of the output JSON file containing port definitions (DEFAULT: module_outputs.json)',
        type=str,
        default="module_outputs.json",
    )
    args = parser.parse_args()

    # Open the file and retrieve the input_dict.
    try:
        with open(args.input_filename) as f:
            input_dict = json.load(f)
        logging.info('Processing the `{}` input file'.format(args.input_filename))
    except FileNotFoundError:
        logging.error("Failed to open the `{}` file".format(args.input_filename))
        exit(-1)

    # Instantiate Neural Factory - on CPU.
    _ = nemo.core.NeuralModuleFactory(placement=nemo.core.DeviceType.CPU)

    # Instantiate module.
    module = instantiate_module(input_dict)

    # Process inputs.
    inputs = {}
    for inp in input_dict["inputs"]:
        # Special case: Tensor.
        if inp["type"] == "Tensor":
            inputs[inp["name"]] = torch.FloatTensor(inp["value"])
        else:
            inputs[inp["name"]] = inp["value"]

    with torch.no_grad():
        outputs = module(force_pt=True, inputs=inputs["inputs"])

    # Process outputs
    output_dict = []
    for key, val in zip(module.output_ports.keys(), outputs):
        t = type(val).__name__
        # Special case: Tensor.
        if t == "Tensor":
            v = val.numpy().tolist()
        else:
            v = val

        output_dict.append({"name": key, "value": v, "type": t})

    # Generate output filename - for default add prefix based on module name.
    output_filename = (
        args.output_filename
        if args.output_filename != "module_outputs.json"
        else input_dict["name"].lower() + "_outputs.json"
    )
    # Export to JSON.
    with open(output_filename, 'w') as outfile:
        json.dump(output_dict, outfile)

    logging.info("=" * 80)
    logging.info("Finished processing of inputs/outputs the `{}` module: \n{}".format(input_dict["name"], output_dict))
    logging.info("Results exported to `{}`.".format(output_filename))


if __name__ == '__main__':
    get_module_outputs()
