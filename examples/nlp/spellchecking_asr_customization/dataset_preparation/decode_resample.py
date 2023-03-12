# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


import argparse
import json
import multiprocessing
import os

import sox
import tqdm
from sox import Transformer
from tqdm.contrib.concurrent import process_map

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', required=True, type=str, help='path to the original manifest')
parser.add_argument("--num_workers", default=multiprocessing.cpu_count(), type=int, help="Workers to process dataset.")
parser.add_argument(
    "--destination_folder", required=True, type=str, help="Destination folder where audio files will be stored"
)
args = parser.parse_args()


def process(x):
    if not isinstance(x['text'], str):
        x['text'] = ''
    else:
        x['text'] = x['text'].lower().strip()
    _, file_with_ext = os.path.split(x['audio_filepath'])
    name, ext = os.path.splitext(file_with_ext)
    output_wav_path = args.destination_folder + "/" + name + '.wav'
    tfm = Transformer()
    tfm.rate(samplerate=16000)
    tfm.channels(n_channels=1)
    tfm.build(input_filepath=x['audio_filepath'], output_filepath=output_wav_path)
    # x['duration'] = sox.file_info.duration(output_wav_path)
    x['audio_filepath'] = output_wav_path
    return x


def load_data(manifest):
    data = []
    with open(manifest, 'r') as f:
        for line in tqdm.tqdm(f):
            item = json.loads(line)
            data.append(item)
    return data


data = load_data(args.manifest)

data_new = process_map(process, data, max_workers=args.num_workers, chunksize=100)

with open(args.manifest.replace('.json', '_decoded.json'), 'w') as f:
    for item in tqdm.tqdm(data_new):
        f.write(json.dumps(item) + '\n')
