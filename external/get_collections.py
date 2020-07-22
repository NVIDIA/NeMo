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


import os
import json
import argparse

import nemo
from nemo.utils import logging

# For now - hardcoded list.
import nemo.collections.common as col_common
import nemo.collections.asr as col_asr
import nemo.collections.cv as col_cv
import nemo.collections.nlp as col_nlp
import nemo.collections.tts as col_tts



def analyse_collection(id, col):
    return {
        "id": id,
        "name": col.__name__,
        "description": col.__description__,
        "version": col.__version__,
        "author": col.__author__
    }



def main():
    # Parse filename.
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', help='Name of the output JSON file', type= str, default="collections.json")
    args = parser.parse_args()

    # Get collections directory.
    colletions_dir = os.path.dirname(nemo.collections.__file__)
    logging.info('Analysing collections in `{}`'.format(colletions_dir))

    # Get list dir.
    #for sub_dir in os.listdir(colletions_dir):
    #    print(sub_dir)

    col_list = []
    # Iterate over all collections.
    for id, col in zip(["asr", "common", "cv", "nlp", "tts"],[col_asr, col_common, col_cv, col_nlp, col_tts]):
        col_list.append(analyse_collection(id, col))

    # Export to JSON.
    with open(args.filename, 'w') as outfile:
        json.dump(col_list, outfile)

    logging.info('Exported collections to `{}`.'.format(args.filename))

if __name__ == '__main__':
    main()