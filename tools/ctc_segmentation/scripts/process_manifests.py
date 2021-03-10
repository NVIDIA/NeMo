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

import argparse
import json
import os
from pathlib import Path
from random import randint
from typing import List

parser = argparse.ArgumentParser(description="Combine manifests and extract samples")
parser.add_argument(
    "--manifests_dir",
    default='_high_score_manifest.json',
    type=str,
    required=True,
    help='Path to directory with *_high_score_manifest.json files',
)
parser.add_argument(
    "--output_dir", default='output', required=True, help='Path to the directory to store final manifest'
)
parser.add_argument(
    "--num_samples",
    default=0,
    type=int,
    help='The number of samples to get from the beginning, end and randomly from the middle of highly scored samples',
)


def get_samples(manifest: str, num_samples: int) -> List[str]:
    """
    Samples num_samples from the beginning, end and randomly from the middle of the manifest

    Args:
        manifest: path to manifest
        num_samples: number of samples to extract from the beginning, end and randomly from the middle of the manifest

    Returns:
        num_samples: manifest lines
    """
    samples = []
    with open(manifest, 'r', encoding='utf8') as f:
        lines = f.readlines()

    # first n samples
    samples.extend(lines[:num_samples])
    # random in the middle
    for i in range(num_samples):
        samples.append(lines[randint(num_samples, len(lines) - num_samples)])
    # last n samples
    samples.extend(lines[-num_samples:])
    return samples


if __name__ == '__main__':
    args = parser.parse_args()
    manifest_dir = Path(args.manifests_dir)
    manifests = sorted(list(manifest_dir.glob('*_high_score_manifest.json')))
    sample_manifest = os.path.join(args.output_dir, 'sample_manifest.json')
    all_manifest = os.path.join(args.output_dir, 'all_manifest.json')

    if args.num_samples > 0:
        with open(sample_manifest, 'w', encoding='utf8') as f:
            for manifest in manifests:
                samples = get_samples(manifest, args.num_samples)
                for sample in samples:
                    f.write(sample)

        print(f'Sample manifest is saved at {sample_manifest}')

    total_duration = 0
    with open(all_manifest, 'w') as out_f:
        for manifest in manifests:
            with open(manifest, 'r', encoding='utf8') as f:
                for line in f:
                    info = json.loads(line)
                    total_duration += info['duration']
                    out_f.write(line)

    print(f'Aggregated manifest is saved at {all_manifest}')
    print(f'Total files duration: ~{round(total_duration/60)} min or ~{round(total_duration/60/60)} hr')

    sets = {'_high_score_manifest.json': 0, '_low_score_manifest.json': 0, '_del_manifest.json': 0}

    for k in sets:
        files = manifest_dir.glob(f'*{k}')
        for manifest_f in files:
            with open(manifest_f, 'r') as f:
                for line in f:
                    line = json.loads(line)
                    sets[k] += line['duration']

    total = 0
    for k in sets:
        sets[k] = round(sets[k] / 3600, 1)
        total += sets[k]
        name = k.split('_')[1].upper()
        print(f'Total {name} score duration: ~{sets[k]} hr')
    print(f'Total duration: ~{total} hr')
    print('Done.')
