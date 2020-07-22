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

""" Script responsible for generation of a JSON file with list of NeMo collections. """

import argparse
import json
import os

import nemo
from nemo.utils import logging

# List of collections.
collections = {}
try:
    import nemo.collections.asr as col_asr

    collections["asr"] = col_asr
except ModuleNotFoundError as e:
    logging.warning("Collection `asr` not found")

try:
    import nemo.collections.common as col_common

    collections["common"] = col_common
except ModuleNotFoundError as e:
    logging.warning("Collection `common` not found")

try:
    import nemo.collections.cv as col_cv

    collections["cv"] = col_cv
except ModuleNotFoundError as e:
    logging.warning("Collection `cv` not found")

try:
    import nemo.collections.nlp as col_nlp

    collections["nlp"] = col_nlp
except ModuleNotFoundError as e:
    logging.warning("Collection `nlp` not found")

try:
    import nemo.collections.tts as col_tts

    collections["tts"] = col_tts
except ModuleNotFoundError as e:
    logging.warning("Collection `tts` not found")


def analyse_collection(id, col):
    """ Helper function analysing the collection.
    
    Args:
        id: (short) name of the collectino
        col: a collection (module).
    """
    logging.info("  * Processed `{}`".format(id))
    return {
        "id": id,
        "name": col.__name__,
        "description": col.__description__,
        "version": col.__version__,
        "author": col.__author__,
    }


def main():
    """ Main function generating a JSON file with list of NeMo collections. """
    # Parse filename.
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', help='Name of the output JSON file', type=str, default="collections.json")
    args = parser.parse_args()

    # Get collections directory.
    colletions_dir = os.path.dirname(nemo.collections.__file__)
    logging.info('Analysing collections in `{}`'.format(colletions_dir))

    col_list = []
    # Iterate over all collections.
    for id, col in collections.items():
        col_list.append(analyse_collection(id, col))

    # Export to JSON.
    with open(args.filename, 'w') as outfile:
        json.dump(col_list, outfile)

    logging.info('Exported collections to `{}`.'.format(args.filename))


if __name__ == '__main__':
    main()
