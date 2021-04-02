# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
"""
Takes the text in the given manifest and uses it to create a .txt transcript file corresponding to each
.wav file for the MFA library to find.
"""

import argparse
import json
import sys
import unidecode

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True, default=None, type=str)
parser.add_argument("--skip_basic_normalization", action='store_true')
args = parser.parse_args()

def main():
    with open(args.manifest, 'r') as manifest:
        for line in open(args.manifest, 'r'):
            fields = json.loads(line)
            txt_path = fields['audio_filepath'][:-3] + 'txt'
            text = fields['text']
            if not args.skip_basic_normalization:
                text = unidecode.unidecode(text).lower()
            with open(txt_path, 'w') as f_out:
                f_out.write(text)

if __name__ == '__main__':
    main()
