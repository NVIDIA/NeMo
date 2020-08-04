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


""" Script responsible for retrieving module input/output ports.

Exemplary call:
    nemo-get-module-ports -i module_def.json

Args:
    Format of the input JSON file (passed as -i, --input_filename):
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

Required fields: "id", "name", "arguments"

Returns:
    Format of the output JSON file (indicated  as -o, --output_filename):
    {
        "input_ports": {
            "inputs": "axes: (batch, dimension:3, height:224, width:224); elements_type: ImageValue"
        },
        "output_ports": {
            "outputs": "axes: (batch, any:10); elements_type: LogitsType"
        }
    }
"""


import argparse
import importlib
import json

import nemo
from nemo.utils import logging


def instantiate_module(module_dict):
    """
    Raises: Key error in case 
    """
    # Check the required keys.
    for key in ["id", "name", "arguments"]:
        if not key in module_dict.keys():
            logging.error("Loaded file doesn't contain the required `{}` key".format(key))
            exit(-2)

    name = module_dict["name"]
    id = module_dict["id"]
    arguments = module_dict["arguments"]

    # Get class  and module from the "full specification".
    class_name = id.rsplit('.', 1)[1]
    module_name = id.rsplit('.', 1)[0]

    logging.info(' * Trying to instantiace `{}` (`{}`) neural module from `{}`'.format(name, class_name, module_name))

    # Import module.
    module_ = importlib.import_module(module_name)
    # Get class
    class_ = getattr(module_, class_name)

    # Process arguments.
    module_args = {}
    for kv in arguments:
        module_args[kv["name"]] = kv["value"]

    # Instantiate object by passing the arguments.
    module = class_(**module_args)
    return module


def get_module_ports():
    """ Main function retrieving module input/output ports. """
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
        help='Name of the output JSON file containing port definitions (DEFAULT: module_ports.json)',
        type=str,
        default="module_ports.json",
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

    # Retrieve ports.
    input_ports = {k: str(v) for k, v in module.input_ports.items()}
    output_ports = {k: str(v) for k, v in module.output_ports.items()}

    output_dict = {
        # "name": input_dict["name"],
        # "id": input_dict["id"],
        "input_ports": input_ports,
        "output_ports": output_ports,
    }

    # Generate output filename - for default add prefix based on module name.
    output_filename = (
        args.output_filename
        if args.output_filename != "module_ports.json"
        else input_dict["name"].lower() + "_ports.json"
    )
    # Export to JSON.
    with open(output_filename, 'w') as outfile:
        json.dump(output_dict, outfile)

    logging.info("=" * 80)
    logging.info(
        "Finished analysis of inputs/output ports the `{}` module: \n{}".format(input_dict["name"], output_dict)
    )
    logging.info("Results exported to `{}`.".format(output_filename))


if __name__ == '__main__':
    get_module_ports()
