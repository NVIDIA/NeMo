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
The script converts raw text to the NeMo format for punctuation and capitalization task.

"""

import argparse
import os

from get_tatoeba_data import create_text_and_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for punctuation and capitalization tasks')
    parser.add_argument("-s", "--source_file", required=True, type=str, help="Path to the source file")
    parser.add_argument("-o", "--output_dir", required=True, type=str, help="Path to the output directory")
    args = parser.parse_args()

    if not os.path.exists(args.source_file):
        raise ValueError(f'{args.source_file} was not found')

    os.makedirs(args.output_dir, exist_ok=True)
    create_text_and_labels(args.output_dir, args.source_file)

    print(f'Processing of the {args.source_file} is complete')
