# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

""" Script responsible for generation of JSON file containing a list of modules. """

import argparse
import inspect
import json
import os

import nemo
from nemo.utils import logging


def analyze_member(name, obj, module_list):
    """ Helper function analysing the passed object and, if ok, adding a record to module list.
    
    Args:
        name: name of the member
        obj: member (class/function etc.)
        module_list: list of modules that (probably) will be expanded.
    """
    # It is not a class - skip it.
    if not inspect.isclass(obj):
        return

    # Check inheritance - we know that all our datasets/modules/losses inherit from Serialization,
    # Btw. Serialization is also required by this script.
    if not issubclass(obj, nemo.core.Serialization):
        return

    logging.info("  * Processed `{}`".format(str(obj)))

    module_list.append(
        {
            "name": name,
            "cls": str(obj),
            # Temporary solution: mockup arguments.
            "arguments": [
                "jasper",
                "activation",
                "feat_in",
                "normalization_mode",
                "residual_mode",
                "norm_groups",
                "conv_mask",
                "frame_splicing",
                "init_mode",
            ],
            # Temporary solution: mockup input types.
            "input_types": {
                "audio_signal": "axes: (batch, dimension, time); elements_type: MelSpectrogramType",
                "length": "axes: (batch,); elements_type: LengthType",
            },
            # Temporary solution: mockup output types.
            "output_types": {
                "encoder_output": "axes: (batch, dimension, time); elements_type: AcousticEncodedRepresentation"
            },
        }
    )


def main():
    """ Main function analysing the indicated NeMo collection and generating a JSON file with module descriptions. """
    # Parse filename.
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', help='ID of the collection', type=str)
    parser.add_argument('--filename', help='Name of the output JSON file', type=str, default="modules.json")
    args = parser.parse_args()

    # Load the right collection.
    if args.collection == "asr":
        import nemo.collections.asr as col
    elif args.collection == "common":
        import nemo.collections.common as col
    elif args.collection == "cv":
        import nemo.collections.cv as col
    elif args.collection == "nlp":
        import nemo.collections.nlp as col
    elif args.collection == "tts":
        import nemo.collections.tts as col
    else:
        logging.error("Coudn't process the incidated `{}` collection".format(args.collection))
        logging.info("Please indicate one of the existing collections using `--collection [asr|commom|cv|nlp|tts]`")
        exit(-1)

    module_list = []
    # Iterate over the packages in the indicated collection.
    logging.info("Analysing the `{}` collection".format(args.collection))

    try:  # Datasets in dataset folder
        logging.info("Analysing the 'data' package")
        for name, obj in inspect.getmembers(col.data):
            analyze_member(name, obj, module_list)
    except AttributeError as e:
        logging.info("  * No datasets found")

    try:  # Datasets in dataset folder
        logging.info("Analysing the 'datasets' package")
        for name, obj in inspect.getmembers(col.datasets):
            analyze_member(name, obj, module_list)
    except AttributeError as e:
        logging.info("  * No datasets found")

    try:  # Modules
        logging.info("Analysing the 'modules' package")
        for name, obj in inspect.getmembers(col.modules):
            analyze_member(name, obj, module_list)
    except AttributeError as e:
        logging.info("  * No modules found")

    try:  # Losses
        logging.info("Analysing the 'losses' package")
        for name, obj in inspect.getmembers(col.losses):
            analyze_member(name, obj, module_list)
    except AttributeError as e:
        logging.info("  * No losses found")

    # Export to JSON.
    with open(args.filename, 'w') as outfile:
        json.dump(module_list, outfile)

    logging.info(
        'Finished analysis of the `{}` collection, results exported to `{}`.'.format(args.collection, args.filename)
    )


if __name__ == '__main__':
    main()
