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

""" Script responsible for analysis the provided graph (modules and connections) and returning their status. 

Args:
    Format of the input JSON file (passed as --input_filename):
    {
        "modules": [
            {
                "name": "cifar100_dl",
                "id": "nemo.collections.cv.modules.data_layers.cifar100_datalayer.CIFAR100DataLayer",
                "module_type": "datalayer",
                "arguments": [
                    {
                        "name": "height",
                        "value": 224
                    },
                    {
                        "name": "width",
                        "value": 224
                    }
                ]
            },
            {
                "name": "my_image_encoder",
                "id": "nemo.collections.cv.modules.trainables.image_encoder.ImageEncoder",
                "module_type": "trainable",
                "arguments": [
                    {
                        "name": "model_type",
                        "value": "resnet50"
                    },
                    {
                        "name": "output_size",
                        "value": 10
                    },
                    {
                        "name": "return_feature_maps",
                        "value": false
                    }
                ]
            }
        ],
        "connections": [
            "cifar100_dl.images->my_image_encoder.inputs",
            "cifar100_dl.coarse_labels->my_image_encoder.inputs"
        ]
    }

Required fields: "modules", "connections"

Returns:
    Format of the output JSON file (indicated  as --output_filename):
    {
        "cifar100_dl.images->my_image_encoder.inputs": "NeuralTypeComparisonResult.SAME",
        "cifar100_dl.coarse_labels->my_image_encoder.inputs": "NeuralTypeComparisonResult.INCOMPATIBLE"
    }
"""

import argparse
import importlib
import json

from scripts.integration.get_module_ports import instantiate_module

import nemo
from nemo.utils import logging


def get_module_port(module_str: str):
    """ Function returns module name and port. """
    lst = module_str.split(".")
    # Handle two cases.
    if len(lst) == 2:
        # Handle module.port case.
        # E.g. "audiotomelspectrogrampreprocessor0.processed_signal"
        module_name = lst[0]
        module_port = lst[1]
    elif len(lst) == 3:
        # Handle step.module.port case.
        # E.g. "0.audiotomelspectrogrampreprocessor0.processed_signal"
        module_name = lst[1]
        module_port = lst[2]
    else:
        raise KeyError("Connection `{}` has improper format".format(connection_str))
    return module_name, module_port


def validate_connections():
    """ Main function analysing the provided graph (modules and connections) and returning their status. """
    # Parse filename.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_filename',
        '-i',
        help='Name of the input JSON file containing graph definition (modules and connections) to be validated',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--output_filename',
        '-o',
        help='Name of the output JSON file containing statuses (DEFAULT: statuses.json)',
        type=str,
        default="statuses.json",
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

    # Check the required keys.
    for key in ["modules", "connections"]:
        if not key in input_dict.keys():
            logging.error("Loaded file doesn't contain the required `{}` key".format(key))
            exit(-2)

    # Instantiate modules.
    modules = {}
    for module_dict in input_dict["modules"]:
        modules[module_dict["name"]] = instantiate_module(module_dict)

    # Analyse the connections.
    output_dict = {}
    for con_dict in input_dict["connections"]:
        # Split connection.
        producer, consumer = con_dict.split("->")
        prod_name, prod_port = get_module_port(producer)
        cons_name, cons_port = get_module_port(consumer)

        print(prod_name, " : ", prod_port, " -> ", cons_name, " : ", cons_port)

        # Compare definitions.
        prod_type = modules[prod_name].output_ports[prod_port]
        cons_type = modules[cons_name].input_ports[cons_port]
        status = prod_type.compare(cons_type)

        output_dict[con_dict] = str(status)

    # Output filename.
    output_filename = args.output_filename

    # Export to JSON.
    with open(output_filename, 'w') as outfile:
        json.dump(output_dict, outfile)

    logging.info("=" * 80)
    outputs = "".join("* {}: {}\n".format(k, v) for k, v in output_dict.items())
    logging.info('Finished analysis of {} connections:\n{}'.format(len(input_dict["connections"]), outputs))
    logging.info('Results exported to `{}`.'.format(output_filename))


if __name__ == '__main__':
    validate_connections()
