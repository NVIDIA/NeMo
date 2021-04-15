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
#

import argparse
import logging
import os
import subprocess
import sys

parser = argparse.ArgumentParser(
    description='Train an n-gram language model with KenLM to be used with BeamSearch decoder of ASR models.'
)
parser.add_argument("--input_path", required=True, type=str)
parser.add_argument("--output_path", required=True, type=str)
parser.add_argument("--ngram_length", required=True, type=int)
parser.add_argument("--kenlm_path", required=True, type=str)

args = parser.parse_args()

""" LMPLZ ARGUMENT SETUP """
args = [
    os.path.join(args.kenlm_path, 'lmplz'),
    "-o",
    args.ngram_length,
    "--text",
    args.input_path,
    "--arpa",
    f"{args.output_path}.arpa",
    "--discount_fallback",
]

result = subprocess.run(args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr)

""" BINARY BUILD """
args = [os.path.join(args.kenlm_path, "build_binary"), "trie", f"{args.output_path}.arpa", args.output_path]

logging.info(f"Running binary_build command \n\n{' '.join(args)}\n\n")

result = subprocess.run(args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr)
