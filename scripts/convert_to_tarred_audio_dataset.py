# Copyright 2020 NVIDIA. All Rights Reserved.
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
# This script converts an existing audio dataset with a manifest to
# a tarred and sharded audio dataset that can be read by the
# TarredAudioToTextDataLayer.

import argparse
import json
import os
import random
import tarfile

parser = argparse.ArgumentParser(
    description="Convert an existing ASR dataset to tarballs compatible with TarredAudioToTextDataLayer."
)
parser.add_argument(
    "--manifest_path", default=None, type=str, required=True, help="Path to the existing dataset's manifest."
)

# Optional arguments
parser.add_argument(
    "--target_dir",
    default='./tarred',
    type=str,
    help="Target directory for resulting tarballs and manifest. Defaults to `./tarred`. Creates the path if ncessary.",
)
parser.add_argument(
    "--num_shards",
    default=1,
    type=int,
    help="Number of shards (tarballs) to create. Used for partitioning data among workers.",
)
parser.add_argument(
    "--shuffle",
    action='store_true',
    help="Whether or not to randomly shuffle the samples in the manifest before tarring/sharding.",
)
args = parser.parse_args()


def create_shard(entries, target_dir, new_entries, shard_id):
    """Creates a tarball containing the audio files from `entries`.
    """
    tar = tarfile.open(os.path.join(target_dir, f'audio_{shard_id}.tar'), mode='w')

    for entry in entries:
        # We squash the filename since we do not preserve directory structure of audio files in the tarball.
        base, ext = os.path.splitext(entry['audio_filepath'])
        base = base.replace('/', '_')
        # Need the following replacement as long as WebDataset splits on first period
        base = base.replace('.', '_')
        squashed_filename = f'{base}{ext}'
        tar.add(entry['audio_filepath'], arcname=squashed_filename)

        new_entry = {
            'audio_filepath': squashed_filename,
            'duration': entry['duration'],
            'text': entry['text'],
            'shard_id': shard_id,  # Keep shard ID for recordkeeping
        }
        new_entries.append(new_entry)

    tar.close()


def main():
    manifest_path = args.manifest_path
    target_dir = args.target_dir
    num_shards = args.num_shards
    shuffle = args.shuffle

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Read the existing manifest
    entries = []
    with open(manifest_path, 'r') as m:
        for line in m:
            entries.append(json.loads(line))

    if shuffle:
        print("Shuffling...")
        random.shuffle(entries)

    # Create shards and updated manifest entries
    new_entries = []
    for i in range(num_shards):
        start_idx = (len(entries) // num_shards) * i
        end_idx = start_idx + (len(entries) // num_shards)
        if i == num_shards - 1:
            end_idx = len(entries)  # Last shard gets the leftovers.
        print(f"Shard {i} will have {end_idx - start_idx} entries.")

        create_shard(entries[start_idx:end_idx], target_dir, new_entries, i)

    # Write manifest
    new_manifest_path = os.path.join(target_dir, 'tarred_audio_manifest.json')
    with open(new_manifest_path, 'w') as m2:
        for entry in new_entries:
            json.dump(entry, m2)
            m2.write('\n')


if __name__ == "__main__":
    main()
