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

"""
# This script converts an existing audio dataset with a manifest to
# a tarred and sharded audio dataset that can be read by the
# TarredAudioToTextDataLayer.

# Please make sure your audio_filepath DOES NOT CONTAIN '-sub'!
# Because we will use it to handle files which have duplicate filenames but with different offsets
# (see function create_shard for details)


# Bucketing can help to improve the training speed. You may use --buckets_num to specify the number of buckets.
# It creates multiple tarred datasets, one per bucket, based on the audio durations.
# The range of [min_duration, max_duration) is split into equal sized buckets.
# Recommend to use --sort_in_shards to speedup the training by reducing the paddings in the batches
# More info on how to use bucketing feature: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/datasets.html

# Usage:
1) Creating a new tarfile dataset

python convert_to_tarred_audio_dataset.py \
    --manifest_path=<path to the manifest file> \
    --target_dir=<path to output directory> \
    --num_shards=<number of tarfiles that will contain the audio> \
    --max_duration=<float representing maximum duration of audio samples> \
    --min_duration=<float representing minimum duration of audio samples> \
    --shuffle --shuffle_seed=1 \
    --sort_in_shards \
    --workers=-1
"""

import argparse
import json
import os
import random
import tarfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from hydra.utils import instantiate
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.tts.parts.preprocessing.features import FeatureReader
from nemo.collections.tts.parts.utils.tarred_dataset_utils import get_file_id
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    filter_dataset_by_duration,
    get_audio_filepaths
)
from nemo.utils import logging


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert an existing TTS dataset to an equivalent tarred dataset."
    )
    parser.add_argument("--output_dir", type=Path, help="Output directory to store tarred dataset in.", required=True)
    parser.add_argument(
        "--manifest_path", type=Path, required=True, help="Path to the existing dataset's manifest."
    )
    parser.add_argument("--audio_dir", type=Path, required=True, help="Base directory where audio is stored.")
    parser.add_argument(
        "--feature_config_path", type=Path, required=True, help="Path to feature config file.",
    )
    parser.add_argument("--feature_dir", type=Path, help="Path to directory where feature data was stored.")
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker processes')
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Number of shards (tarballs) to create. Used for partitioning data among workers.",
    )
    parser.add_argument(
        "--num_buckets", type=int, default=1, help="Number of buckets to create based on duration.",
    )
    parser.add_argument(
        '--min_duration',
        type=float,
        default=0.0,
        help='Minimum duration of audio clip in the dataset.',
    )
    parser.add_argument(
        '--max_duration',
        type=float,
        required=True,
        help='Maximum duration of audio clip in the dataset.',
    )
    parser.add_argument(
        "--shard_manifests",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Write sharded manifests along with the aggregated manifest.",
    )
    parser.add_argument(
        "--shuffle",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether or not to randomly shuffle the samples in the manifest before tarring/sharding.",
    )
    parser.add_argument("--shuffle_seed", type=int, default=100, help="Random seed for use if shuffling is enabled.")
    parser.add_argument(
        "--sort_in_shards",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether or not to sort samples inside the shards based on their duration.",
    )
    args = parser.parse_args()

    return args


@dataclass
class TTSTarredDatasetConfig:
    min_duration: float
    max_duration: float
    num_shards: int
    shuffle: bool
    shuffle_seed: Optional[int]
    sort_in_shards: bool
    shard_manifests: bool


@dataclass
class TTSTarredDatasetMetadata:
    created_datetime: Optional[str] = None
    version: int = 0
    num_samples_per_shard: Optional[int] = None
    dataset_config: Optional[TTSTarredDatasetConfig] = None
    history: Optional[List[Any]] = field(default_factory=lambda: [])

    def __post_init__(self):
        self.created_datetime = self.get_current_datetime()

    def get_current_datetime(self):
        return datetime.now().strftime("%m-%d-%Y %H-%M-%S")

    @classmethod
    def from_file(cls, filepath: str):
        config = OmegaConf.load(filepath)
        metadata = cls()
        metadata.__dict__.update(**config)
        return metadata


class TTSTarredDatasetBuilder:
    """
    Helper class that constructs a tarred dataset from scratch, or concatenates tarred datasets
    together and constructs manifests for them.
    """

    def __init__(self, config: TTSTarredDatasetConfig):
        self.config = config
        self.file_counts = defaultdict(int)
        self.lock = threading.Lock()

    def create_new_dataset(
        self,
        output_dir: Path,
        manifest_path: Path,
        audio_dir: Path,
        feature_readers: List[FeatureReader],
        feature_dir: Optional[Path],
        num_workers: int
    ):
        """
        Creates a new tarred dataset from a given manifest file.

        Args:
            output_dir: Output directory.
            manifest_path: Path to the original TTS manifest.
            audio_dir: Directory containing audio files.
            feature_dir: Directory containing feature files, optional
            num_workers: Integer denoting number of parallel worker processes which will write tarfiles.

        Output:
            Writes tarfiles, along with the tarred dataset compatible manifest file.
            Also preserves a record of the metadata used to construct this tarred dataset.
        """
        if self.config is None:
            raise ValueError("Config has not been set. Please call `configure(config: TTSTarredDatasetConfig)`")

        output_dir.mkdir(parents=True, exist_ok=True)

        config = self.config

        # Read the existing manifest
        entries = read_manifest(manifest=manifest_path)
        filtered_entries, total_hours, filtered_hours = filter_dataset_by_duration(
            entries=entries,
            min_duration=config.min_duration,
            max_duration=config.max_duration
        )

        logging.info(f"Original # of files: {len(entries)}")
        logging.info(f"Filtered # of files: {len(filtered_entries)}")
        logging.info(f"Original duration: {total_hours:.2f} hours")
        logging.info(f"Filtered duration: {filtered_hours:.2f} hours")

        if len(filtered_entries) == 0:
            raise ValueError("Found no data to create tarred dataset with after filtering.")

        if config.shuffle:
            print("Shuffling...")
            random.seed(config.shuffle_seed)
            random.shuffle(entries)

        # Create shards and updated manifest entries
        print(f"Number of samples added : {len(entries)}")
        print(f"Remainder: {len(entries) % config.num_shards}")

        start_indices = []
        end_indices = []
        # Build indices
        for i in range(config.num_shards):
            start_idx = (len(entries) // config.num_shards) * i
            end_idx = start_idx + (len(entries) // config.num_shards)
            #print(f"Shard {i} has entries {start_idx} ~ {end_idx}")
            files = set()
            for ent_id in range(start_idx, end_idx):
                files.add(entries[ent_id]["audio_filepath"])
            #print(f"Shard {i} contains {len(files)} files")
            if i == config.num_shards - 1:
                # We discard in order to have the same number of entries per shard.
                print(f"Have {len(entries) - end_idx} entries left over that will be discarded.")

            start_indices.append(start_idx)
            end_indices.append(end_idx)

        with Parallel(n_jobs=num_workers, backend='threading', verbose=config.num_shards) as parallel:
            # Call parallel tarfile construction
            new_entries_list = parallel(
                delayed(self._create_shard)(
                    output_dir=output_dir,
                    shard_id=i,
                    entries=entries[start_idx:end_idx],
                    audio_dir=audio_dir,
                    feature_readers=feature_readers,
                    feature_dir=feature_dir,
                )
                for i, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices))
            )

        if config.shard_manifests:
            sharded_manifests_dir = output_dir / 'sharded_manifests'
            if not os.path.exists(sharded_manifests_dir):
                os.makedirs(sharded_manifests_dir)

            for manifest in new_entries_list:
                shard_id = manifest[0]['shard_id']
                new_manifest_shard_path = os.path.join(sharded_manifests_dir, f'manifest_{shard_id}.json')
                with open(new_manifest_shard_path, 'w', encoding='utf-8') as m2:
                    for entry in manifest:
                        json.dump(entry, m2)
                        m2.write('\n')

        # Flatten the list of list of entries to a list of entries
        new_entries = [sample for manifest in new_entries_list for sample in manifest]
        del new_entries_list

        print("Total number of entries in manifest :", len(new_entries))

        # Write manifest
        new_manifest_path = output_dir / manifest_path.parts[-1]
        with open(new_manifest_path, 'w', encoding='utf-8') as m2:
            for entry in new_entries:
                json.dump(entry, m2)
                m2.write('\n')

        # Write metadata (default metadata for new datasets)
        num_samples_per_shard = len(new_entries) // config.num_shards
        metadata = TTSTarredDatasetMetadata(
            dataset_config=config,
            num_samples_per_shard=num_samples_per_shard
        )
        # Write metadata
        metadata_yaml = OmegaConf.structured(metadata)
        metadata_path = output_dir / 'metadata.yaml'
        OmegaConf.save(metadata_yaml, metadata_path, resolve=False)

    def _create_shard(
        self,
        output_dir: Path,
        shard_id: int,
        entries: List[Dict[str, any]],
        audio_dir: Path,
        feature_readers: List[FeatureReader],
        feature_dir: Optional[Path],
    ):
        """Creates a tarball containing the audio files from `entries`.
        """
        if self.config.sort_in_shards:
            entries.sort(key=lambda x: x["duration"], reverse=False)

        out_entries = []
        shard_output_path = output_dir / f'audio_{shard_id}.tar'
        with tarfile.open(shard_output_path, mode='w', dereference=True) as tar_file:
            for entry in entries:
                audio_filepath_abs, audio_filepath_rel = get_audio_filepaths(manifest_entry=entry, audio_dir=audio_dir)

                if not audio_filepath_abs.exists():
                    raise FileNotFoundError(f"Could not find file: {audio_filepath_abs}")

                file_id = get_file_id(audio_filepath_rel)

                with self.lock:
                    file_count = self.file_counts[file_id]
                    self.file_counts[file_id] += 1

                if file_count == 0:
                    base_filename = file_id
                else:
                    base_filename = f"{file_id}-sub{self.file_counts[file_id]}"

                for feature_reader in feature_readers:
                    feature_filepath = feature_reader.get_feature_filepath(
                        manifest_entry=entry,
                        audio_dir=audio_dir,
                        feature_dir=feature_dir,
                    )
                    tar_suffix = feature_reader.get_tarred_suffix(feature_filepath)
                    tar_filename = f"{base_filename}.{tar_suffix}"
                    if 'offset' not in entry:
                        tar_file.add(name=feature_filepath, arcname=tar_filename)
                    else:
                        bytes_io = feature_reader.serialize(
                            manifest_entry=entry, audio_dir=audio_dir, feature_dir=feature_dir
                        )
                        bytes_io.seek(0)
                        tar_info = tarfile.TarInfo(name=tar_filename)
                        tar_info.size = bytes_io.getbuffer().nbytes
                        tar_info.mtime = time.time()
                        tar_file.addfile(tarinfo=tar_info, fileobj=bytes_io)

                out_entry = {
                    'audio_filepath': str(audio_filepath_rel),
                    'shard_id': shard_id,  # Keep shard ID for record keeping
                    'duration': entry['duration'],
                }

                if 'normalized_text' in entry:
                    out_entry['text'] = entry['normalized_text']
                elif 'text' in entry:
                    out_entry['text'] = entry['text']

                if 'speaker' in entry:
                    out_entry['speaker'] = entry['speaker']

                if 'offset' in entry:
                    out_entry['offset'] = entry['offset']

                out_entries.append(out_entry)

        return out_entries


def create_tar_datasets(
    output_dir: Path,
    manifest_path: Path,
    audio_dir: Path,
    feature_readers: List[FeatureReader],
    feature_dir: Optional[Path],
    num_workers: int,
    shard_manifests: bool,
    num_shards: int,
    min_duration: float,
    max_duration: float,
    shuffle: bool,
    shuffle_seed: Optional[int],
    sort_in_shards: bool
):
    # Create a tarred dataset from scratch
    config = TTSTarredDatasetConfig(
        max_duration=max_duration,
        min_duration=min_duration,
        num_shards=num_shards,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        sort_in_shards=sort_in_shards,
        shard_manifests=shard_manifests
    )
    builder = TTSTarredDatasetBuilder(config=config)
    builder.create_new_dataset(
        output_dir=output_dir,
        manifest_path=manifest_path,
        audio_dir=audio_dir,
        feature_readers=feature_readers,
        feature_dir=feature_dir,
        num_workers=num_workers
    )


def main():
    args = get_args()

    output_dir = args.output_dir
    manifest_path = args.manifest_path
    audio_dir = args.audio_dir
    feature_dir = args.feature_dir
    feature_config_path = args.feature_config_path
    num_workers = args.num_workers
    num_shards = args.num_shards
    num_buckets = args.num_buckets
    min_duration = args.min_duration
    max_duration = args.max_duration
    shard_manifests = args.shard_manifests
    shuffle = args.shuffle
    sort_in_shards = args.sort_in_shards

    if shuffle:
        shuffle_seed = args.shuffle_seed
    else:
        shuffle_seed = None

    assert num_shards > 0
    assert num_buckets > 0
    assert 0.0 <= min_duration < max_duration
    assert manifest_path.exists()
    assert audio_dir.exists()

    feature_config = OmegaConf.load(feature_config_path)
    feature_reader_dict = instantiate(feature_config.feature_readers)
    feature_readers = feature_reader_dict.values()

    if num_buckets == 1:
        create_tar_datasets(
            output_dir=output_dir,
            manifest_path=manifest_path,
            audio_dir=audio_dir,
            feature_readers=feature_readers,
            feature_dir=feature_dir,
            num_workers=num_workers,
            num_shards=num_shards,
            min_duration=min_duration,
            max_duration=max_duration,
            shard_manifests=shard_manifests,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            sort_in_shards=sort_in_shards
        )
    else:
        bucket_length = (max_duration - min_duration) / float(num_buckets)
        for i in range(num_buckets):
            bucket_num = i + 1
            output_dir_i = output_dir / f"bucket{bucket_num}"
            min_duration_i = min_duration + (i * bucket_length)
            max_duration_i = min_duration + ((i + 1) * bucket_length)
            if i == (num_buckets - 1):
                # add a small number to cover the samples with exactly duration of max_duration in the last bucket.
                max_duration_i += 1e-5

            print(f"Creating bucket {bucket_num} in {output_dir_i} with "
                  f"min_duration={min_duration_i} and max_duration={max_duration_i}")
            create_tar_datasets(
                output_dir=output_dir_i,
                manifest_path=manifest_path,
                audio_dir=audio_dir,
                feature_readers=feature_readers,
                feature_dir=feature_dir,
                num_workers=num_workers,
                shard_manifests=shard_manifests,
                num_shards=num_shards,
                min_duration=min_duration_i,
                max_duration=max_duration_i,
                shuffle=shuffle,
                shuffle_seed=shuffle_seed,
                sort_in_shards=sort_in_shards
            )
            print(f"Created bucket {bucket_num}.")


if __name__ == "__main__":
    main()
