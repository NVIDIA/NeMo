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
import gc
import json
import os

import text_utils
import torch
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import nemo.collections.asr as nemo_asr

CHUNK_SIZE = 8192
CHUNK_BUFFER_SIZE = 512
TOKEN_OFFSET = 100


def main():
    parser = argparse.ArgumentParser(
        description='Preprocessing samples for training a KenLM model for BPE based ASR models.'
    )
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--input_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--do_lowercase", action='store_true')
    args = parser.parse_args()

    """ TOKENIZER SETUP """
    model = nemo_asr.models.EncDecCTCModelBPE.restore_from(args.model_path, map_location=torch.device('cpu'))

    """ DATASET SETUP """
    dataset = text_utils.read_text(args.input_path, lowercase=args.do_lowercase)
    text_utils.tokenize_text(
        dataset,
        model.tokenizer,
        path=args.output_path,
        chunk_size=CHUNK_SIZE,
        buffer_size=CHUNK_BUFFER_SIZE,
        token_offset=TOKEN_OFFSET,
    )


if __name__ == '__main__':
    main()
