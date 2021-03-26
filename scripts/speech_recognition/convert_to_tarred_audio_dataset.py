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
"""
# This script converts an existing audio dataset with a manifest to
# a tarred and sharded audio dataset that can be read by the
# TarredAudioToTextDataLayer.

# Please make sure your audio_filepath DOES NOT CONTAIN '-sub'!
# Because we will use it to handle files which have duplicate filenames but with different offsets
# (see function create_shard for details)

# Usage:
1) Creating a new tarfile dataset

python convert_to_tarred_audio_dataset.py \
    --manifest_path=<path to the manifest file> \
    --target_dir=<path to output directory> \
    --num_shards=<number of tarfiles that will contain the audio>
    --max_duration=<float representing maximum duration of audio samples> \
    --min_duration=<float representing minimum duration of audio samples> \
    --shuffle --shuffle_seed=1


2) Concatenating more tarfiles to a pre-existing tarred dataset

python convert_to_tarred_audio_dataset.py \
    --manifest_path=<path to the tarred manifest file> \
    --metadata_path=<path to the metadata.yaml (or metadata_version_{X}.yaml) file> \
    --target_dir=<path to output directory where the original tarfiles are contained> \
    --max_duration=<float representing maximum duration of audio samples> \
    --min_duration=<float representing minimum duration of audio samples> \
    --shuffle --shuffle_seed=1 \
    --concat_manifest_paths \
    <space seperated paths to 1 or more manifest files to concatenate into the original tarred dataset>

3) Writing an empty metadata file

python convert_to_tarred_audio_dataset.py \
    --target_dir=<path to output directory> \
    # any other optional argument
    --num_shards=8 \
    --max_duration=16.7 \
    --min_duration=0.01 \
    --shuffle \
    --shuffle_seed=1 \
    --write_metadata

"""
import argparse
import copy
import json
import os
import random
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional

from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf, open_dict

parser = argparse.ArgumentParser(
    description="Convert an existing ASR dataset to tarballs compatible with TarredAudioToTextDataLayer."
)
parser.add_argument(
    "--manifest_path", default=None, type=str, required=False, help="Path to the existing dataset's manifest."
)

parser.add_argument(
    '--concat_manifest_paths',
    nargs='+',
    default=None,
    type=str,
    required=False,
    help="Path to the additional dataset's manifests that will be concatenated with base dataset.",
)

# Optional arguments
parser.add_argument(
    "--target_dir",
    default='./tarred',
    type=str,
    help="Target directory for resulting tarballs and manifest. Defaults to `./tarred`. Creates the path if necessary.",
)

parser.add_argument(
    "--metadata_path", required=False, default=None, type=str, help="Path to metadata file for the dataset.",
)

parser.add_argument(
    "--num_shards",
    default=-1,
    type=int,
    help="Number of shards (tarballs) to create. Used for partitioning data among workers.",
)
parser.add_argument(
    '--max_duration',
    default=None,
    type=float,
    help='Maximum duration of audio clip in the dataset. By default, it is None and will not filter files.',
)
parser.add_argument(
    '--min_duration',
    default=None,
    type=float,
    help='Minimum duration of audio clip in the dataset. By default, it is None and will not filter files.',
)
parser.add_argument(
    "--shuffle",
    action='store_true',
    help="Whether or not to randomly shuffle the samples in the manifest before tarring/sharding.",
)
parser.add_argument("--shuffle_seed", type=int, help="Random seed for use if shuffling is enabled.")
parser.add_argument(
    '--write_metadata',
    action='store_true',
    help=(
        "Flag to write a blank metadata with the current call config. "
        "Note that the metadata will not contain the number of shards, "
        "and it must be filled out by the user."
    ),
)
parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
args = parser.parse_args()


@dataclass
class ASRTarredDatasetConfig:
    num_shards: int = -1
    shuffle: bool = False
    max_duration: Optional[float] = None
    min_duration: Optional[float] = None
    shuffle_seed: Optional[int] = None


@dataclass
class ASRTarredDatasetMetadata:
    created_datetime: Optional[str] = None
    version: int = 0
    num_samples_per_shard: Optional[int] = None
    is_concatenated_manifest: bool = False

    dataset_config: Optional[ASRTarredDatasetConfig] = ASRTarredDatasetConfig()
    history: Optional[List[Any]] = field(default_factory=lambda: [])

    def __post_init__(self):
        self.created_datetime = self.get_current_datetime()

    def get_current_datetime(self):
        return datetime.now().strftime("%m-%d-%Y %H-%M-%S")

    @classmethod
    def from_config(cls, config: DictConfig):
        obj = cls()
        obj.__dict__.update(**config)
        return obj

    @classmethod
    def from_file(cls, filepath: str):
        config = OmegaConf.load(filepath)
        return ASRTarredDatasetMetadata.from_config(config=config)


class ASRTarredDatasetBuilder:
    """
    Helper class that constructs a tarred dataset from scratch, or concatenates tarred datasets
    together and constructs manifests for them.
    """

    def __init__(self):
        self.config = None

    def configure(self, config: ASRTarredDatasetConfig):
        """
        Sets the config generated from command line overrides.

        Args:
            config: ASRTarredDatasetConfig dataclass object.
        """
        self.config = config  # type: ASRTarredDatasetConfig

        if self.config.num_shards < 0:
            raise ValueError("`num_shards` must be > 0. Please fill in the metadata information correctly.")

    def create_new_dataset(self, manifest_path: str, target_dir: str = "./tarred/", num_workers: int = 0):
        """
        Creates a new tarred dataset from a given manifest file.

        Args:
            manifest_path: Path to the original ASR manifest.
            target_dir: Output directory.
            num_workers: Integer denoting number of parallel worker processes which will write tarfiles.
                Defaults to 1 - which denotes sequential worker process.

        Output:
            Writes tarfiles, along with the tarred dataset compatible manifest file.
            Also preserves a record of the metadata used to construct this tarred dataset.
        """
        if self.config is None:
            raise ValueError("Config has not been set. Please call `configure(config: ASRTarredDatasetConfig)`")

        if manifest_path is None:
            raise FileNotFoundError("Manifest filepath cannot be None !")

        config = self.config  # type: ASRTarredDatasetConfig

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Read the existing manifest
        entries, filtered_entries, filtered_duration = self._read_manifest(manifest_path, config)

        if len(filtered_entries) > 0:
            print(f"Filtered {len(filtered_entries)} files which amounts to {filtered_duration} seconds of audio.")

        if config.shuffle:
            random.seed(config.shuffle_seed)
            print("Shuffling...")
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
            print(f"Shard {i} has entries {start_idx} ~ {end_idx}")
            if i == config.num_shards - 1:
                # We discard in order to have the same number of entries per shard.
                print(f"Have {len(entries) - end_idx} entries left over that will be discarded.")

            start_indices.append(start_idx)
            end_indices.append(end_idx)

        with Parallel(n_jobs=num_workers, verbose=config.num_shards) as parallel:
            # Call parallel tarfile construction
            new_entries_list = parallel(
                delayed(self._create_shard)(entries[start_idx:end_idx], target_dir, i)
                for i, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices))
            )

        # Flatten the list of list of entries to a list of entries
        new_entries = [sample for manifest in new_entries_list for sample in manifest]
        del new_entries_list

        print("Total number of files in manifest :", len(new_entries))

        # Write manifest
        new_manifest_path = os.path.join(target_dir, 'tarred_audio_manifest.json')
        with open(new_manifest_path, 'w') as m2:
            for entry in new_entries:
                json.dump(entry, m2)
                m2.write('\n')

        # Write metadata (default metadata for new datasets)
        new_metadata_path = os.path.join(target_dir, 'metadata.yaml')
        metadata = ASRTarredDatasetMetadata()

        # Update metadata
        metadata.dataset_config = config
        metadata.num_samples_per_shard = len(new_entries) // config.num_shards

        # Write metadata
        metadata_yaml = OmegaConf.structured(metadata)
        OmegaConf.save(metadata_yaml, new_metadata_path, resolve=True)

    def create_concatenated_dataset(
        self,
        base_manifest_path: str,
        manifest_paths: List[str],
        metadata: ASRTarredDatasetMetadata,
        target_dir: str = "./tarred_concatenated/",
        num_workers: int = 1,
    ):
        """
        Creates new tarfiles in order to create a concatenated dataset, whose manifest contains the data for
        both the original dataset as well as the new data submitted in manifest paths.

        Args:
            base_manifest_path: Path to the manifest file which contains the information for the original
                tarred dataset (with flattened paths).
            manifest_paths: List of one or more paths to manifest files that will be concatenated with above
                base tarred dataset.
            metadata: ASRTarredDatasetMetadata dataclass instance with overrides from command line.
            target_dir: Output directory

        Output:
            Writes tarfiles which with indices mapping to a "concatenated" tarred dataset,
            along with the tarred dataset compatible manifest file which includes information
            about all the datasets that comprise the concatenated dataset.

            Also preserves a record of the metadata used to construct this tarred dataset.
        """
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        if base_manifest_path is None:
            raise FileNotFoundError("Base manifest filepath cannot be None !")

        if manifest_paths is None or len(manifest_paths) == 0:
            raise FileNotFoundError("List of additional manifest filepaths cannot be None !")

        config = ASRTarredDatasetConfig(**(metadata.dataset_config))

        # Read the existing manifest (no filtering here)
        base_entries, _, _ = self._read_manifest(base_manifest_path, config)
        print(f"Read base manifest containing {len(base_entries)} samples.")

        # Precompute number of samples per shard
        if metadata.num_samples_per_shard is None:
            num_samples_per_shard = len(base_entries) // config.num_shards
        else:
            num_samples_per_shard = metadata.num_samples_per_shard

        print("Number of samples per shard :", num_samples_per_shard)

        # Compute min and max duration and update config (if no metadata passed)
        print(f"Selected max duration : {config.max_duration}")
        print(f"Selected min duration : {config.min_duration}")

        entries = []
        for new_manifest_idx in range(len(manifest_paths)):
            new_entries, filtered_new_entries, filtered_duration = self._read_manifest(
                manifest_paths[new_manifest_idx], config
            )

            if len(filtered_new_entries) > 0:
                print(
                    f"Filtered {len(filtered_new_entries)} files which amounts to {filtered_duration:0.2f}"
                    f" seconds of audio from manifest {manifest_paths[new_manifest_idx]}."
                )

            entries.extend(new_entries)

        if config.shuffle:
            random.seed(config.shuffle_seed)
            print("Shuffling...")
            random.shuffle(entries)

        # Drop last section of samples that cannot be added onto a chunk
        drop_count = len(entries) % num_samples_per_shard
        total_new_entries = len(entries)
        entries = entries[:-drop_count]

        print(
            f"Dropping {drop_count} samples from total new samples {total_new_entries} since they cannot "
            f"be added into a uniformly sized chunk."
        )

        # Create shards and updated manifest entries
        num_added_shards = len(entries) // num_samples_per_shard

        print(f"Number of samples in base dataset : {len(base_entries)}")
        print(f"Number of samples in additional datasets : {len(entries)}")
        print(f"Number of added shards : {num_added_shards}")
        print(f"Remainder: {len(entries) % num_samples_per_shard}")

        start_indices = []
        end_indices = []
        shard_indices = []
        for i in range(num_added_shards):
            start_idx = (len(entries) // num_added_shards) * i
            end_idx = start_idx + (len(entries) // num_added_shards)
            shard_idx = i + config.num_shards
            print(f"Shard {shard_idx} has entries {start_idx + len(base_entries)} ~ {end_idx + len(base_entries)}")

            start_indices.append(start_idx)
            end_indices.append(end_idx)
            shard_indices.append(shard_idx)

        with Parallel(n_jobs=num_workers, verbose=num_added_shards) as parallel:
            # Call parallel tarfile construction
            new_entries_list = parallel(
                delayed(self._create_shard)(entries[start_idx:end_idx], target_dir, shard_idx)
                for i, (start_idx, end_idx, shard_idx) in enumerate(zip(start_indices, end_indices, shard_indices))
            )

        # Flatten the list of list of entries to a list of entries
        new_entries = [sample for manifest in new_entries_list for sample in manifest]
        del new_entries_list

        # Write manifest
        if metadata is None:
            new_version = 1  # start with `1`, where `0` indicates the base manifest + dataset
        else:
            new_version = metadata.version + 1

        print("Total number of files in manifest :", len(base_entries) + len(new_entries))

        new_manifest_path = os.path.join(target_dir, f'tarred_audio_manifest_version_{new_version}.json')
        with open(new_manifest_path, 'w') as m2:
            # First write all the entries of base manifest
            for entry in base_entries:
                json.dump(entry, m2)
                m2.write('\n')

            # Finally write the new entries
            for entry in new_entries:
                json.dump(entry, m2)
                m2.write('\n')

        # Preserve historical metadata
        base_metadata = metadata

        # Write metadata (updated metadata for concatenated datasets)
        new_metadata_path = os.path.join(target_dir, f'metadata_version_{new_version}.yaml')
        metadata = ASRTarredDatasetMetadata()

        # Update config
        config.num_shards = config.num_shards + num_added_shards

        # Update metadata
        metadata.version = new_version
        metadata.dataset_config = config
        metadata.num_samples_per_shard = num_samples_per_shard
        metadata.is_concatenated_manifest = True
        metadata.created_datetime = metadata.get_current_datetime()

        # Attach history
        current_metadata = OmegaConf.structured(base_metadata.history)
        metadata.history = current_metadata

        # Write metadata
        metadata_yaml = OmegaConf.structured(metadata)
        OmegaConf.save(metadata_yaml, new_metadata_path, resolve=True)

    def _read_manifest(self, manifest_path: str, config: ASRTarredDatasetConfig):
        """Read and filters data from the manifest"""
        # Read the existing manifest
        entries = []
        filtered_entries = []
        filtered_duration = 0.0
        with open(manifest_path, 'r') as m:
            for line in m:
                entry = json.loads(line)
                if (config.max_duration is None or entry['duration'] < config.max_duration) and (
                    config.min_duration is None or entry['duration'] > config.min_duration
                ):
                    entries.append(entry)
                else:
                    filtered_entries.append(entry)
                    filtered_duration += entry['duration']

        return entries, filtered_entries, filtered_duration

    def _create_shard(self, entries, target_dir, shard_id):
        """Creates a tarball containing the audio files from `entries`.
        """
        new_entries = []
        tar = tarfile.open(os.path.join(target_dir, f'audio_{shard_id}.tar'), mode='w')

        count = dict()
        for entry in entries:
            # We squash the filename since we do not preserve directory structure of audio files in the tarball.
            base, ext = os.path.splitext(entry['audio_filepath'])
            base = base.replace('/', '_')
            # Need the following replacement as long as WebDataset splits on first period
            base = base.replace('.', '_')
            squashed_filename = f'{base}{ext}'
            if squashed_filename not in count:
                tar.add(entry['audio_filepath'], arcname=squashed_filename)

            if 'label' in entry:
                base, ext = os.path.splitext(squashed_filename)
                # no suffix if it's single sample or starting sub parts, -sub1 for the second subpart -sub2 -sub3 ,etc.
                if squashed_filename not in count:
                    to_write = squashed_filename
                    count[squashed_filename] = 1
                else:
                    to_write = base + "-sub" + str(count[squashed_filename]) + ext
                    count[squashed_filename] += 1

                new_entry = {
                    'audio_filepath': to_write,
                    'duration': entry['duration'],
                    'text': entry['text'],
                    'label': entry['label'],
                    'offset': entry['offset'],
                    'shard_id': shard_id,  # Keep shard ID for recordkeeping
                }
            else:
                count[squashed_filename] = 1
                new_entry = {
                    'audio_filepath': squashed_filename,
                    'duration': entry['duration'],
                    'text': entry['text'],
                    'shard_id': shard_id,  # Keep shard ID for recordkeeping
                }

            new_entries.append(new_entry)

        tar.close()
        return new_entries

    @classmethod
    def setup_history(cls, base_metadata: ASRTarredDatasetMetadata, history: List[Any]):
        if 'history' in base_metadata.keys():
            for history_val in base_metadata.history:
                cls.setup_history(history_val, history)

        if base_metadata is not None:
            metadata_copy = copy.deepcopy(base_metadata)
            with open_dict(metadata_copy):
                metadata_copy.pop('history', None)
            history.append(metadata_copy)


def main():
    manifest_path = args.manifest_path
    concat_manifest_paths = args.concat_manifest_paths
    target_dir = args.target_dir
    metadata_path = args.metadata_path
    num_shards = args.num_shards
    max_duration = args.max_duration
    min_duration = args.min_duration
    shuffle = args.shuffle
    seed = args.shuffle_seed if args.shuffle_seed else None
    write_metadata = args.write_metadata
    num_workers = args.workers

    builder = ASRTarredDatasetBuilder()

    if write_metadata:
        metadata = ASRTarredDatasetMetadata()
        dataset_cfg = ASRTarredDatasetConfig(
            num_shards=num_shards,
            shuffle=shuffle,
            max_duration=max_duration,
            min_duration=min_duration,
            shuffle_seed=seed,
        )
        metadata.dataset_config = dataset_cfg

        output_path = os.path.join(target_dir, 'default_metadata.yaml')
        OmegaConf.save(metadata, output_path, resolve=True)
        print(f"Default metadata written to {output_path}")
        exit(0)

    if concat_manifest_paths is None or len(concat_manifest_paths) == 0:
        print("Creating new tarred dataset ...")

        # Create a tarred dataset from scratch
        config = ASRTarredDatasetConfig(
            num_shards=num_shards,
            shuffle=shuffle,
            max_duration=max_duration,
            min_duration=min_duration,
            shuffle_seed=seed,
        )
        builder.configure(config)
        builder.create_new_dataset(manifest_path=manifest_path, target_dir=target_dir, num_workers=num_workers)

    else:
        print("Concatenating multiple tarred datasets ...")

        # Implicitly update config from base details
        if metadata_path is not None:
            metadata = ASRTarredDatasetMetadata.from_file(metadata_path)
        else:
            raise ValueError("`metadata` yaml file path must be provided!")

        # Preserve history
        history = []
        builder.setup_history(OmegaConf.structured(metadata), history)
        metadata.history = history

        # Add command line overrides (everything other than num_shards)
        metadata.dataset_config.max_duration = max_duration
        metadata.dataset_config.min_duration = min_duration
        metadata.dataset_config.shuffle = shuffle
        metadata.dataset_config.shuffle_seed = seed

        builder.configure(metadata.dataset_config)

        # Concatenate a tarred dataset onto a previous one
        builder.create_concatenated_dataset(
            base_manifest_path=manifest_path,
            manifest_paths=concat_manifest_paths,
            metadata=metadata,
            target_dir=target_dir,
            num_workers=num_workers,
        )


if __name__ == "__main__":
    main()
